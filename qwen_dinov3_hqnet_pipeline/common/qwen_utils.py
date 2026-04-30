from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple


def _optional_import_torch():
    import torch  # noqa: F401

    return torch


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected yaml mapping in {path}")
    return data


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def resolve_nested_attr(obj: Any, dotted_name: str) -> Any:
    cur = obj
    for part in dotted_name.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def resolve_vision_module(model: Any) -> Tuple[Any, str]:
    candidates = (
        "visual",
        "vision_tower",
        "vision_model",
        "model.visual",
        "model.vision_tower",
        "model.vision_model",
        "model.model.visual",
        "model.model.vision_tower",
    )
    for name in candidates:
        module = resolve_nested_attr(model, name)
        if module is not None:
            return module, name
    raise AttributeError("Unable to locate a Qwen vision module on the loaded model.")


def extract_hidden_size(module: Any) -> Optional[int]:
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for key in ("hidden_size", "embed_dim", "dim"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    for attr_name in ("hidden_size", "embed_dim", "dim"):
        value = getattr(module, attr_name, None)
        if isinstance(value, int) and value > 0:
            return int(value)
    return None


def extract_patch_size(module: Any) -> int:
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for key in ("patch_size", "spatial_patch_size"):
            value = getattr(cfg, key, None)
            if isinstance(value, int) and value > 0:
                return int(value)
    patch_embed = getattr(module, "patch_embed", None)
    if patch_embed is not None:
        value = getattr(patch_embed, "patch_size", None)
        if isinstance(value, int):
            return max(1, int(value))
        if isinstance(value, Sequence) and value:
            return max(1, int(value[0]))
    return 14


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        valid = set(inspect.signature(fn).parameters.keys())
    except Exception:
        return kwargs
    return {k: v for k, v in kwargs.items() if k in valid}


def _extract_tensor_from_output(output: Any) -> Any:
    torch = _optional_import_torch()
    if torch.is_tensor(output):
        return output
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            if torch.is_tensor(item):
                return item
    hidden_states = getattr(output, "hidden_states", None)
    if isinstance(hidden_states, (list, tuple)) and hidden_states:
        return hidden_states[-1]
    for key in ("last_hidden_state", "image_embeds", "image_hidden_states", "vision_hidden_states"):
        value = getattr(output, key, None)
        if value is None and isinstance(output, dict):
            value = output.get(key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value:
            value = value[-1]
        if torch.is_tensor(value):
            return value
    return None


def load_qwen_vl_model(
    model_name_or_path: str,
    adapter_path: str = "",
    torch_dtype: Optional[str] = "bfloat16",
    device: str = "cpu",
    merge_lora: bool = False,
):
    torch = _optional_import_torch()
    from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

    try:
        from transformers import AutoModelForVision2Seq
    except Exception:
        AutoModelForVision2Seq = None

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        None: torch.float32,
    }[torch_dtype]
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    has_vision = bool(getattr(cfg, "vision_config", None))
    model = None
    if has_vision and AutoModelForVision2Seq is not None:
        try:
            model = AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except Exception:
            model = None
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()
    model.to(device)
    model.eval()
    return model, processor


def build_qwen_image_grid(batch: Any, patch_size: int = 14) -> Any:
    torch = _optional_import_torch()
    if batch.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W] batch, got shape={tuple(batch.shape)}")
    bsz, _, height, width = batch.shape
    return torch.tensor(
        [[1, max(1, height // patch_size), max(1, width // patch_size)] for _ in range(bsz)],
        dtype=torch.long,
        device=batch.device,
    )


def prepare_processor_image_inputs(processor: Any, image: Any) -> Dict[str, Any]:
    if hasattr(processor, "image_processor"):
        batch = processor.image_processor(images=image, return_tensors="pt")
        if isinstance(batch, dict):
            return dict(batch)
    batch = processor(text=[""], images=[image], return_tensors="pt")
    return {k: v for k, v in batch.items() if k in ("pixel_values", "image_grid_thw")}


def extract_qwen_vision_features(model: Any, pixel_values: Any, image_grid_thw: Optional[Any] = None) -> Any:
    torch = _optional_import_torch()
    vision_module, _ = resolve_vision_module(model)
    patch_size = extract_patch_size(vision_module)
    if image_grid_thw is None:
        image_grid_thw = build_qwen_image_grid(pixel_values, patch_size=patch_size)

    for fn_name in ("forward", "__call__"):
        fn = getattr(vision_module, fn_name, None)
        if fn is None:
            continue
        kwargs = _filter_kwargs(
            fn,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "grid_thw": image_grid_thw,
                "output_hidden_states": True,
                "return_dict": True,
            },
        )
        try:
            output = fn(**kwargs)
            tensor = _extract_tensor_from_output(output)
            if tensor is not None and torch.is_tensor(tensor):
                return tensor
        except Exception:
            continue

    for fn_name in ("get_image_features", "extract_image_features"):
        fn = getattr(model, fn_name, None)
        if fn is None:
            continue
        kwargs = _filter_kwargs(
            fn,
            {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "output_hidden_states": True,
                "return_dict": True,
            },
        )
        try:
            output = fn(**kwargs)
            tensor = _extract_tensor_from_output(output)
            if tensor is not None and torch.is_tensor(tensor):
                return tensor
        except Exception:
            continue

    raise RuntimeError("Failed to extract vision features from the loaded Qwen-VL model.")


def token_sequence_to_feature_map(tokens: Any, image_size: Optional[Tuple[int, int]] = None, patch_size: int = 14) -> Any:
    torch = _optional_import_torch()
    if not torch.is_tensor(tokens):
        raise TypeError("Expected tensor tokens.")
    if tokens.ndim == 4:
        return tokens
    if tokens.ndim != 3:
        raise ValueError(f"Expected [B,N,C] or [B,C,H,W], got shape={tuple(tokens.shape)}")

    batch, num_tokens, channels = tokens.shape
    if image_size is not None:
        height = max(1, image_size[0] // patch_size)
        width = max(1, image_size[1] // patch_size)
        if height * width == num_tokens - 1:
            tokens = tokens[:, 1:, :]
            num_tokens = tokens.shape[1]
        if height * width != num_tokens:
            side = int(num_tokens**0.5)
            height = side
            width = max(1, num_tokens // max(1, side))
    else:
        if int((num_tokens - 1) ** 0.5) ** 2 == num_tokens - 1:
            tokens = tokens[:, 1:, :]
            num_tokens = tokens.shape[1]
        side = int(num_tokens**0.5)
        height = side
        width = max(1, num_tokens // max(1, side))

    usable = height * width
    tokens = tokens[:, :usable, :]
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


def build_pyramid_from_feature_map(feature_map: Any, num_outs: int = 4) -> Tuple[Any, ...]:
    import torch.nn.functional as F

    if feature_map.ndim != 4:
        raise ValueError(f"Expected [B,C,H,W], got shape={tuple(feature_map.shape)}")
    feats = [feature_map]
    for _ in range(max(0, num_outs - 1)):
        feats.append(F.avg_pool2d(feats[-1], kernel_size=2, stride=2, ceil_mode=True))
    return tuple(feats[:num_outs])
