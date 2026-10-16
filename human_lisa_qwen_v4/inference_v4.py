from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from human_lisa_qwen_v1.inference_v1 import (
    suppress_near_duplicate_person_records,
)
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone
from human_lisa_qwen_v1.official_eval_v1 import encode_binary_mask

from .collator_v4 import _find_last_subsequence
from .guided_decode_v4 import (
    build_guided_prefix_allowed_tokens_fn,
    mean_logprob_by_spans,
    person_object_spans,
)
from .modeling_v4 import Stage1V4Model, require_record_token_states
from .schema_v4 import (
    PROMPT_V4,
    apply_chat_template_v4,
    canonicalize_person_ids_v4,
    validate_prediction_object_v4,
)
from .token_rows_v4 import add_v4_special_tokens
from .train_stage1_v4 import hidden_size_from_config


def build_chat_prefix_v4(processor: Any) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": PROMPT_V4},
            ],
        }
    ]
    return apply_chat_template_v4(processor, messages)


def decode_generated_json_v4(
    tokenizer: Any, generated_ids: torch.Tensor
) -> Tuple[str, bool, int]:
    ids = [int(value) for value in generated_ids.tolist()]
    eos_ids = {
        int(value)
        for value in [
            getattr(tokenizer, "eos_token_id", None),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        ]
        if value is not None and int(value) >= 0
    }
    stop = next(
        (index for index, token_id in enumerate(ids) if token_id in eos_ids),
        len(ids),
    )
    eos_seen = stop < len(ids)
    content_ids = ids[:stop]
    text = tokenizer.decode(
        content_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return text.strip(), eos_seen, len(content_ids)


def _find_manifest(checkpoint: Path) -> Tuple[Path, Dict[str, Any]]:
    candidates = [
        checkpoint / "manifest.json",
        checkpoint.parent / "manifest.json",
        checkpoint.parent.parent / "manifest.json",
    ]
    for path in candidates:
        if path.is_file():
            return path, json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(
        f"cannot find a v4 manifest near checkpoint {checkpoint}"
    )


def _restore_named_parameters(
    model: torch.nn.Module, state: Dict[str, torch.Tensor], label: str
) -> None:
    parameters = dict(model.named_parameters())
    missing = sorted(set(state) - set(parameters))
    if missing:
        raise RuntimeError(f"{label} has unknown parameter names: {missing[:8]}")
    for name, value in state.items():
        target = parameters[name]
        if target.shape != value.shape:
            raise RuntimeError(
                f"{label} shape mismatch for {name}: "
                f"{tuple(value.shape)} vs {tuple(target.shape)}"
            )
        target.data.copy_(value.to(device=target.device, dtype=target.dtype))


def load_stage1_v4_checkpoint(
    checkpoint_dir: str,
    *,
    base_model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> Tuple[Stage1V4Model, Any, Dict[str, Any]]:
    checkpoint = Path(checkpoint_dir)
    manifest_path, manifest = _find_manifest(checkpoint)
    base_model_path = base_model_path or str(manifest["model"])
    processor = AutoProcessor.from_pretrained(
        checkpoint / "processor", trust_remote_code=True
    )
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    base = load_qwen_vl_backbone(base_model_path, dtype)
    token_ids = add_v4_special_tokens(processor, base)
    expected_token_ids = {
        name: int(value) for name, value in manifest["token_ids"].items()
    }
    if token_ids != expected_token_ids:
        raise RuntimeError(
            f"processor/checkpoint token ids differ: {token_ids} vs "
            f"{expected_token_ids}"
        )
    from peft import PeftModel

    profile = str(manifest.get("profile", "vision_tuned"))
    if profile == "vision_tuned":
        base = PeftModel.from_pretrained(
            base,
            checkpoint / "base_or_adapter",
            is_trainable=False,
        )
    task_state_path = checkpoint / "stage1_v4_tasks.pt"
    task_state = torch.load(
        task_state_path, map_location="cpu", weights_only=False
    )
    decoder = manifest.get("decoder") or {}
    decoder_dim = int(task_state.get("decoder_dim", decoder.get("dim", 256)))
    decoder_heads = int(
        task_state.get("decoder_heads", decoder.get("heads", 8))
    )
    decoder_layers = int(
        task_state.get("decoder_layers", decoder.get("layers", 2))
    )
    config = getattr(base, "config", None)
    vision_config = getattr(config, "vision_config", None)
    wrapper = Stage1V4Model(
        base_model=base,
        hidden_size=hidden_size_from_config(base),
        token_ids=token_ids,
        image_token_id=int(getattr(config, "image_token_id", 248056)),
        visual_merge_size=int(
            task_state.get(
                "visual_merge_size",
                getattr(vision_config, "spatial_merge_size", 2),
            )
        ),
        decoder_dim=decoder_dim,
        decoder_heads=decoder_heads,
        decoder_layers=decoder_layers,
        text_task_weights=task_state.get("text_task_weights"),
        component_weights=task_state.get("component_weights"),
        task_weights=task_state.get("task_weights"),
    )
    wrapper.load_task_state(str(task_state_path))
    merger_path = checkpoint / "visual_merger.pt"
    if merger_path.is_file():
        merger_state = torch.load(
            merger_path, map_location="cpu", weights_only=True
        )
        _restore_named_parameters(wrapper, merger_state, "visual merger")
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    wrapper.to(device).eval()
    resolved_manifest = dict(manifest)
    resolved_manifest["manifest_path"] = str(manifest_path)
    resolved_manifest["checkpoint"] = str(checkpoint)
    return wrapper, processor, resolved_manifest


def _boxes_from_object_v4(
    obj: Dict[str, Any], *, device: torch.device
) -> torch.Tensor:
    values = [person["box"] for person in obj["persons"]]
    return torch.tensor(
        values, device=device, dtype=torch.float32
    ).reshape(-1, 4) / 1000.0


def semantic_health_v4(
    obj: Dict[str, Any],
    *,
    boxes: torch.Tensor,
    pose_coordinates: torch.Tensor,
    mask_logits: torch.Tensor,
    person_scores: torch.Tensor,
    eos_seen: bool,
    raw_token_count: int,
    max_new_tokens: int,
) -> Dict[str, Any]:
    people = len(obj["persons"])
    errors: List[str] = []
    if boxes.shape != (people, 4):
        errors.append("box_count_mismatch")
    if pose_coordinates.shape != (people, 17, 2):
        errors.append("pose_count_mismatch")
    if mask_logits.shape[0] != people:
        errors.append("seg_count_mismatch")
    if person_scores.shape != (people,):
        errors.append("person_score_count_mismatch")
    if not eos_seen:
        errors.append("missing_eos")
    if raw_token_count >= int(max_new_tokens):
        errors.append("generation_hit_token_cap")
    for name, value in {
        "boxes": boxes,
        "pose": pose_coordinates,
        "masks": mask_logits,
        "scores": person_scores,
    }.items():
        if not torch.isfinite(value).all():
            errors.append(f"nonfinite_{name}")
    if boxes.numel() and (
        (boxes[:, 2] <= boxes[:, 0]).any()
        or (boxes[:, 3] <= boxes[:, 1]).any()
        or (boxes < 0).any()
        or (boxes > 1).any()
    ):
        errors.append("invalid_box_geometry")
    duplicate_pairs = 0
    for first in range(people):
        for second in range(first + 1, people):
            top_left = torch.maximum(boxes[first, :2], boxes[second, :2])
            bottom_right = torch.minimum(boxes[first, 2:], boxes[second, 2:])
            intersection = (bottom_right - top_left).clamp(min=0).prod()
            first_area = (boxes[first, 2:] - boxes[first, :2]).prod()
            second_area = (boxes[second, 2:] - boxes[second, :2]).prod()
            union = (first_area + second_area - intersection).clamp(min=1e-7)
            if float(intersection / union) >= 0.9:
                duplicate_pairs += 1
    if duplicate_pairs:
        errors.append("duplicate_boxes")
    return {
        "ok": not errors,
        "errors": errors,
        "persons": people,
        "duplicate_pairs": duplicate_pairs,
        "eos_seen": bool(eos_seen),
        "raw_token_count": int(raw_token_count),
    }


@torch.no_grad()
def decode_dense_for_text_v4(
    model: Stage1V4Model,
    processor: Any,
    *,
    image: Any,
    raw_text: str,
    obj: Dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    prefix = build_chat_prefix_v4(processor)
    eos = processor.tokenizer.eos_token or ""
    target_text = raw_text + eos
    processor.tokenizer.padding_side = "right"
    inputs = processor(
        text=[prefix + target_text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    target_ids = [
        int(value)
        for value in processor.tokenizer(
            target_text, add_special_tokens=False
        )["input_ids"]
    ]
    raw_encoding = processor.tokenizer(
        raw_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    raw_ids = [int(value) for value in raw_encoding["input_ids"]]
    sequence = [int(value) for value in inputs["input_ids"][0].tolist()]
    target_start = _find_last_subsequence(sequence, target_ids)
    if target_start < 1:
        raise RuntimeError("cannot locate generated JSON in multimodal sequence")
    if len(sequence) > 5000:
        raise RuntimeError(
            f"generated dense decode exceeds max_length=5000: {len(sequence)}"
        )
    if sequence[target_start : target_start + len(raw_ids)] != raw_ids:
        raise RuntimeError("generated JSON token ids do not round-trip")

    with torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=(device.type == "cuda" and dtype == torch.bfloat16),
    ):
        outputs = model.base_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        people = len(obj["persons"])
        pose_states = require_record_token_states(
            input_ids=inputs["input_ids"],
            hidden=hidden,
            labels=None,
            token_id=model.token_ids["pose"],
            expected_counts=[people],
            task_name="pose",
            start_positions=[target_start],
        )[0]
        seg_states = require_record_token_states(
            input_ids=inputs["input_ids"],
            hidden=hidden,
            labels=None,
            token_id=model.token_ids["seg"],
            expected_counts=[people],
            task_name="seg",
            start_positions=[target_start],
        )[0]
        visual_tokens, spatial_shape = model._visual_tokens_and_shape(
            inputs["input_ids"], hidden, inputs["image_grid_thw"], 0
        )
        pose_coordinates, pose_visibility_logits = model.pose_decoder(
            pose_states, visual_tokens, spatial_shape=spatial_shape
        )
        mask_logits = model.mask_decoder(
            seg_states, visual_tokens, spatial_shape=spatial_shape
        )

    if raw_ids:
        logits = outputs.logits[
            0,
            target_start - 1 : target_start - 1 + len(raw_ids),
        ].float()
        targets = torch.tensor(raw_ids, device=logits.device, dtype=torch.long)
        token_logprobs = (
            logits.gather(1, targets[:, None]).squeeze(1)
            - torch.logsumexp(logits, dim=-1)
        ).detach().cpu().tolist()
    else:
        token_logprobs = []
    spans = person_object_spans(raw_text)
    mean_logprobs = mean_logprob_by_spans(
        processor.tokenizer,
        raw_ids,
        token_logprobs,
        spans,
    )
    if len(mean_logprobs) != len(obj["persons"]):
        raise RuntimeError(
            "person span/record mismatch: "
            f"spans={len(mean_logprobs)} people={len(obj['persons'])}"
        )
    person_scores = torch.tensor(
        [math.exp(max(-30.0, min(0.0, value))) for value in mean_logprobs],
        device=device,
        dtype=torch.float32,
    )
    return {
        "boxes": _boxes_from_object_v4(obj, device=device),
        "pose_coordinates": pose_coordinates,
        "pose_visibility_logits": pose_visibility_logits,
        "mask_logits": mask_logits,
        "person_scores": person_scores,
        "person_mean_logprobs": torch.tensor(
            mean_logprobs, device=device, dtype=torch.float32
        ),
    }


def assemble_coco_records_v4(
    *,
    image_id: int,
    width: int,
    height: int,
    obj: Dict[str, Any],
    dense: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    boxes = dense["boxes"].detach().float().cpu()
    pose = dense["pose_coordinates"].detach().float().cpu()
    visibility = dense["pose_visibility_logits"].float().softmax(dim=-1)
    visibility = visibility[..., 1:].sum(dim=-1).detach().cpu()
    masks = dense["mask_logits"].detach().float()
    scores = dense["person_scores"].detach().float().cpu()
    logprobs = dense["person_mean_logprobs"].detach().float().cpu()
    people = len(obj["persons"])
    if not (
        boxes.shape == (people, 4)
        and pose.shape == (people, 17, 2)
        and masks.shape[0] == people
        and scores.shape == (people,)
    ):
        raise RuntimeError("v4 record and dense prediction counts differ")
    records: List[Dict[str, Any]] = []
    for index, person in enumerate(obj["persons"]):
        x1 = float(boxes[index, 0] * width)
        y1 = float(boxes[index, 1] * height)
        x2 = float(boxes[index, 2] * width)
        y2 = float(boxes[index, 3] * height)
        keypoints: List[float] = []
        for joint in range(17):
            keypoints.extend(
                [
                    float(pose[index, joint, 0] * width),
                    float(pose[index, joint, 1] * height),
                    float(visibility[index, joint]),
                ]
            )
        mask = F.interpolate(
            masks[index : index + 1, None],
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        records.append(
            {
                "image_id": int(image_id),
                "category_id": 1,
                "person_id": int(person["person_id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(scores[index].clamp(1e-12, 1.0)),
                "constant_score": 1.0,
                "person_mean_logprob": float(logprobs[index]),
                "gender": int(person["gender"]),
                "age": float(person["age"]),
                "keypoints": keypoints,
                "segmentation": encode_binary_mask(
                    mask.sigmoid().ge(0.5).cpu().numpy()
                ),
            }
        )
    return records


@torch.no_grad()
def infer_samples_v4(
    model: Stage1V4Model,
    processor: Any,
    samples: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 5000,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    if int(max_new_tokens) != 5000:
        raise ValueError("v4 generated evaluation requires max_new_tokens=5000")
    device = device or next(model.parameters()).device
    prefix = build_chat_prefix_v4(processor)
    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=[prefix] * len(samples),
        images=[sample["image"] for sample in samples],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    prompt_width = int(inputs["input_ids"].shape[1])
    allowed = build_guided_prefix_allowed_tokens_fn(
        processor.tokenizer, prompt_width=prompt_width
    )
    started = time.perf_counter()
    with torch.autocast(
        "cuda",
        dtype=dtype,
        enabled=(device.type == "cuda" and dtype == torch.bfloat16),
    ):
        generation = model.base_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            prefix_allowed_tokens_fn=allowed,
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
    generation_elapsed = time.perf_counter() - started
    results: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        raw, eos_seen, raw_token_count = decode_generated_json_v4(
            processor.tokenizer, generation[index, prompt_width:]
        )
        result: Dict[str, Any] = {
            "image_id": int(sample["id"]),
            "raw": raw,
            "eos_seen": eos_seen,
            "schema_ok": False,
            "raw_schema_ok": False,
            "person_id_repairs": 0,
            "records": [],
            "score_strategy": "generated_person_record_mean_logprob",
            "generation_latency_seconds": generation_elapsed / len(samples),
        }
        try:
            obj = json.loads(raw)
            repairs = canonicalize_person_ids_v4(obj)
            validate_prediction_object_v4(obj)
            dense_started = time.perf_counter()
            dense = decode_dense_for_text_v4(
                model,
                processor,
                image=sample["image"],
                raw_text=raw,
                obj=obj,
                dtype=dtype,
                device=device,
            )
            result["dense_latency_seconds"] = (
                time.perf_counter() - dense_started
            )
            health = semantic_health_v4(
                obj,
                boxes=dense["boxes"],
                pose_coordinates=dense["pose_coordinates"],
                mask_logits=dense["mask_logits"],
                person_scores=dense["person_scores"],
                eos_seen=eos_seen,
                raw_token_count=raw_token_count,
                max_new_tokens=max_new_tokens,
            )
            raw_records = assemble_coco_records_v4(
                image_id=int(sample["id"]),
                width=int(sample["width"]),
                height=int(sample["height"]),
                obj=obj,
                dense=dense,
            )
            records, suppression = suppress_near_duplicate_person_records(
                raw_records, iou_threshold=0.90
            )
            result.update(
                {
                    "schema_ok": True,
                    "raw_schema_ok": repairs == 0,
                    "person_id_repairs": repairs,
                    "health": health,
                    "raw_record_count": len(raw_records),
                    "duplicate_suppression": suppression,
                    "records": records,
                }
            )
        except Exception as error:
            result["error"] = repr(error)
            result["health"] = {
                "ok": False,
                "errors": ["inference_exception"],
            }
        results.append(result)
    return results


def summarize_inference_audit_v4(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    total = len(results)
    denominator = max(1, total)
    return {
        "images": total,
        "schema_ok_rate": sum(
            int(result.get("schema_ok", False)) for result in results
        )
        / denominator,
        "raw_schema_ok_rate": sum(
            int(result.get("raw_schema_ok", False)) for result in results
        )
        / denominator,
        "person_id_repair_image_rate": sum(
            int(int(result.get("person_id_repairs", 0)) > 0)
            for result in results
        )
        / denominator,
        "person_id_repairs": sum(
            int(result.get("person_id_repairs", 0)) for result in results
        ),
        "health_ok_rate": sum(
            int((result.get("health") or {}).get("ok", False))
            for result in results
        )
        / denominator,
        "eos_rate": sum(
            int(result.get("eos_seen", False)) for result in results
        )
        / denominator,
        "inference_error_rate": sum(
            int("error" in result) for result in results
        )
        / denominator,
    }
