import argparse
import inspect
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
except Exception:
    Qwen3_5ForConditionalGeneration = None

from qwen4b_coco_unihuman_ft.dataset import CocoUniHumanSFTDataset, SFTCollator
from qwen4b_coco_unihuman_ft.eval_scheme2 import (
    dump_json,
    evaluate_records_scheme2,
    get_record_gt,
    normalize_struct,
    parse_json_object_from_text,
    parse_tag_struct,
)
from qwen4b_coco_unihuman_ft.prompts import (
    DEFAULT_ASSISTANT_PREFIX_TAG,
    build_vl_chat_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for COCO-UniHuman on Qwen models.")
    parser.add_argument(
        "--config",
        type=str,
        default="qwen4b_coco_unihuman_ft/configs/qwen4b_box_pose_sft.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml config: {path}")
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(cfg: Dict[str, Any]):
    if bool(cfg.get("bf16", False)):
        return torch.bfloat16
    if bool(cfg.get("fp16", False)):
        return torch.float16
    return torch.float32


def build_model(model_name_or_path: str, torch_dtype):
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    has_vision = bool(getattr(cfg, "vision_config", None))
    kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if has_vision and Qwen3_5ForConditionalGeneration is not None:
        try:
            return Qwen3_5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            pass
    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _supports_vision_training(model, processor) -> bool:
    if getattr(processor, "image_processor", None) is None:
        return False
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "vision_config", None) is not None:
        return True
    for attr_name in ("visual", "vision_tower", "vision_model"):
        if hasattr(model, attr_name):
            return True
        inner = getattr(model, "model", None)
        if inner is not None and hasattr(inner, attr_name):
            return True
    return False


def _iter_signature_sources(model):
    seen = set()
    for obj in (model, getattr(model, "base_model", None), getattr(model, "model", None)):
        if obj is None:
            continue
        obj_id = id(obj)
        if obj_id in seen:
            continue
        seen.add(obj_id)
        yield obj


def _model_accepts_vision_inputs(model) -> bool:
    keys = set()
    for obj in _iter_signature_sources(model):
        for fn_name in ("forward", "prepare_inputs_for_generation"):
            fn = getattr(obj, fn_name, None)
            if fn is None:
                continue
            try:
                keys.update(inspect.signature(fn).parameters.keys())
            except Exception:
                continue
    return any(k in keys for k in ("pixel_values", "image_grid_thw", "images"))


def maybe_apply_lora(model, cfg: Dict[str, Any]):
    lora_cfg = cfg.get("lora", {})
    if not isinstance(lora_cfg, dict) or not bool(lora_cfg.get("enable", False)):
        return model

    from peft import LoraConfig, get_peft_model

    target_modules = lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(target_modules, str):
        target_modules_cfg = target_modules
    elif isinstance(target_modules, (list, tuple)):
        target_modules_cfg = list(target_modules)
    else:
        raise ValueError(f"Unsupported lora.target_modules: {target_modules!r}")

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_cfg,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def _build_training_arguments(cfg: Dict[str, Any], has_eval: bool) -> TrainingArguments:
    kwargs: Dict[str, Any] = {
        "output_dir": str(cfg["output_dir"]),
        "per_device_train_batch_size": int(cfg.get("train_batch_size", 1)),
        "per_device_eval_batch_size": int(cfg.get("eval_batch_size", 1)),
        "gradient_accumulation_steps": int(cfg.get("gradient_accumulation_steps", 1)),
        "learning_rate": float(cfg.get("learning_rate", 1e-5)),
        "weight_decay": float(cfg.get("weight_decay", 0.01)),
        "num_train_epochs": float(cfg.get("num_train_epochs", 1)),
        "warmup_ratio": float(cfg.get("warmup_ratio", 0.03)),
        "logging_steps": int(cfg.get("logging_steps", 10)),
        "save_steps": int(cfg.get("save_steps", 500)),
        "eval_steps": int(cfg.get("eval_steps", 500)),
        "save_total_limit": int(cfg.get("save_total_limit", 2)),
        "bf16": bool(cfg.get("bf16", False)),
        "fp16": bool(cfg.get("fp16", False)),
        "dataloader_num_workers": int(cfg.get("dataloader_num_workers", 2)),
        "remove_unused_columns": False,
        "report_to": cfg.get("report_to", []),
        "run_name": str(cfg.get("run_name", os.path.basename(os.path.abspath(str(cfg["output_dir"]))))),
        "logging_first_step": bool(cfg.get("logging_first_step", True)),
    }
    optional_keys = (
        "save_strategy",
        "logging_strategy",
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
    )
    for key in optional_keys:
        if key in cfg:
            kwargs[key] = cfg[key]
    params = inspect.signature(TrainingArguments.__init__).parameters
    eval_value = "steps" if has_eval else "no"
    if "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_value
    elif has_eval and "do_eval" in params:
        kwargs["do_eval"] = True
    return TrainingArguments(**kwargs)


def _jsonable_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        try:
            json.dumps(value)
            out[str(key)] = value
        except Exception:
            out[str(key)] = str(value)
    return out


def _safe_scalar(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return float(out)


def _scalar_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not isinstance(metrics, dict):
        return out
    for key, value in metrics.items():
        scalar = _safe_scalar(value)
        if scalar is not None:
            out[str(key)] = scalar
    return out


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _select_fixed_indices(total: int, count: int, seed: int) -> List[int]:
    if total <= 0 or count <= 0:
        return []
    if count >= total:
        return list(range(total))
    rng = random.Random(int(seed))
    return sorted(rng.sample(range(total), k=int(count)))


def _parse_prediction_text(text: str, image_size: Tuple[int, int]) -> Tuple[Dict[str, Any], str]:
    tag_obj = parse_tag_struct(text, image_size=image_size)
    if tag_obj is not None:
        return normalize_struct(tag_obj, image_size=image_size), "tag"
    json_obj = parse_json_object_from_text(text)
    if json_obj is not None:
        return normalize_struct(json_obj, image_size=image_size), "json"
    return normalize_struct(None, image_size=image_size), "none"


class JsonlMetricCallback(TrainerCallback):
    def __init__(self, output_dir: str) -> None:
        self.output_path = os.path.join(output_dir, "metrics.jsonl")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not getattr(state, "is_world_process_zero", True):
            return
        metrics = _scalar_metrics(logs)
        if not metrics:
            return
        row: Dict[str, Any] = {
            "event": "log",
            "global_step": int(getattr(state, "global_step", 0)),
        }
        if getattr(state, "epoch", None) is not None:
            row["epoch"] = float(state.epoch)
        row.update(metrics)
        _append_jsonl(self.output_path, row)


class SwanLabVisualCallback(TrainerCallback):
    def __init__(
        self,
        cfg: Dict[str, Any],
        processor,
        train_dataset: CocoUniHumanSFTDataset,
        eval_dataset: Optional[CocoUniHumanSFTDataset],
    ) -> None:
        self.cfg = cfg
        self.processor = processor
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.preview_dataset = eval_dataset if eval_dataset is not None else train_dataset
        self.output_dir = str(cfg["output_dir"])
        self.swanlab_cfg = cfg.get("swanlab", {}) if isinstance(cfg.get("swanlab", {}), dict) else {}
        self.stage_eval_cfg = (
            self.swanlab_cfg.get("stage_eval", {})
            if isinstance(self.swanlab_cfg.get("stage_eval", {}), dict)
            else {}
        )
        self._swanlab = None
        self._run = None
        self._stage_indices: Optional[List[int]] = None

    def _is_main_process(self, state) -> bool:
        return bool(getattr(state, "is_world_process_zero", True))

    def _get_swanlab(self):
        if self._swanlab is None:
            try:
                import swanlab
            except Exception as e:
                raise RuntimeError(
                    "SwanLab is enabled in config, but `swanlab` is not installed in the runtime environment."
                ) from e
            self._swanlab = swanlab
        return self._swanlab

    def _ensure_run(self):
        if self._run is not None:
            return self._run
        swanlab = self._get_swanlab()
        init_kwargs: Dict[str, Any] = {
            "project": str(self.swanlab_cfg.get("project", "human-box-pose-attr-sft")),
            "workspace": self.swanlab_cfg.get("workspace", None),
            "experiment_name": str(
                self.swanlab_cfg.get(
                    "experiment_name",
                    os.path.basename(os.path.abspath(self.output_dir)),
                )
            ),
            "description": self.swanlab_cfg.get("description", None),
            "config": _jsonable_cfg(self.cfg),
            "logdir": str(
                self.swanlab_cfg.get(
                    "logdir",
                    os.path.join(self.output_dir, "swanlog"),
                )
            ),
            "mode": str(self.swanlab_cfg.get("mode", "cloud")),
        }
        tags = self.swanlab_cfg.get("tags", None)
        if isinstance(tags, list) and tags:
            init_kwargs["tags"] = [str(tag) for tag in tags]
        init_kwargs = {k: v for k, v in init_kwargs.items() if v not in (None, "", [])}
        self._run = swanlab.init(**init_kwargs)
        return self._run

    def _unwrap_model(self, model):
        return getattr(model, "module", model)

    def _stage_eval_enabled(self) -> bool:
        return bool(self.stage_eval_cfg.get("enable", False))

    def _preview_enabled(self) -> bool:
        return int(self.swanlab_cfg.get("preview_samples", 0)) > 0

    def _get_stage_indices(self) -> List[int]:
        if self._stage_indices is not None:
            return self._stage_indices
        sample_count = int(
            self.stage_eval_cfg.get(
                "samples",
                max(
                    int(self.swanlab_cfg.get("preview_samples", 0)),
                    0,
                ),
            )
        )
        seed = int(self.stage_eval_cfg.get("seed", self.cfg.get("seed", 42)))
        self._stage_indices = _select_fixed_indices(len(self.preview_dataset), sample_count, seed)
        return self._stage_indices

    def _generate_preview_rows(self, model, indices: List[int]) -> Tuple[List[Dict[str, Any]], List[Any]]:
        if not indices:
            return [], []
        swanlab = self._get_swanlab()
        device = next(model.parameters()).device
        tokenizer = getattr(self.processor, "tokenizer", None)
        eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
        pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else eos_token_id
        max_new_tokens = int(self.swanlab_cfg.get("preview_max_new_tokens", 1024))
        preview_limit = min(int(self.swanlab_cfg.get("preview_samples", 0)), len(indices))
        pred_rows: List[Dict[str, Any]] = []
        preview_images: List[Any] = []

        was_training = model.training
        model.eval()
        with torch.no_grad():
            for out_idx, sample_idx in enumerate(indices):
                sample = self.preview_dataset[sample_idx]
                raw_record = self.preview_dataset.records[sample_idx]
                prompt = build_vl_chat_text(
                    processor=self.processor,
                    prompt=str(sample["prompt"]),
                    assistant_prefix=DEFAULT_ASSISTANT_PREFIX_TAG,
                )
                inputs = _prepare_preview_inputs(
                    processor=self.processor,
                    prompt=prompt,
                    image=sample["image"],
                    max_length=int(self.cfg.get("max_length", 4096)),
                )
                inputs = _move_batch_to_device(inputs, device)
                gen_inputs = _filter_generate_inputs(model, inputs)
                gen_ids = model.generate(
                    **gen_inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=pad_token_id,
                    eos_token_id=eos_token_id,
                )
                input_len = gen_inputs["input_ids"].shape[1]
                pred_suffix = self.processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
                pred_text = DEFAULT_ASSISTANT_PREFIX_TAG + pred_suffix
                image_size = sample["image"].size
                pred_struct, parsed_from = _parse_prediction_text(pred_text, image_size=image_size)
                gt_struct = get_record_gt(raw_record, image_size=image_size)
                row = {
                    "id": str(raw_record.get("id", sample["id"])),
                    "image": raw_record.get("image", sample["image_path"]),
                    "image_path": sample["image_path"],
                    "width": int(raw_record.get("width", image_size[0])),
                    "height": int(raw_record.get("height", image_size[1])),
                    "box": raw_record.get("box", []),
                    "pose": raw_record.get("pose", []),
                    "attribute": raw_record.get("attribute", []),
                    "gt_struct": gt_struct,
                    "prediction_text": pred_text,
                    "prediction_struct_post": pred_struct,
                    "parsed_from": parsed_from,
                    "valid_struct": parsed_from != "none",
                }
                pred_rows.append(row)
                if out_idx < preview_limit:
                    caption = (
                        f"id={row['id']}\n"
                        f"parsed_from={parsed_from}\n"
                        f"gt_box={len(gt_struct['box'])}, pred_box={len(pred_struct['box'])}\n\n"
                        f"gt={sample['target_text']}\n\n"
                        f"pred={pred_text}"
                    )
                    preview_images.append(swanlab.Image(sample["image"], caption=caption))
        if was_training:
            model.train()
        return pred_rows, preview_images

    def _run_stage_eval(self, state, model) -> None:
        if not (self._preview_enabled() or self._stage_eval_enabled()):
            return
        indices = self._get_stage_indices()
        if not indices:
            return
        model = self._unwrap_model(model)
        pred_rows, preview_images = self._generate_preview_rows(model, indices)
        if not pred_rows:
            return

        swanlab = self._get_swanlab()
        log_payload: Dict[str, Any] = {}
        if preview_images:
            log_payload["Prediction"] = preview_images

        artifact_dir = Path(self.output_dir) / "stage_eval" / f"step-{int(getattr(state, 'global_step', 0)):08d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pred_jsonl_path = artifact_dir / "predictions.jsonl"
        for row in pred_rows:
            _append_jsonl(str(pred_jsonl_path), row)

        if self._stage_eval_enabled():
            gt_records = [self.preview_dataset.records[idx] for idx in indices]
            metrics = evaluate_records_scheme2(
                gt_records=gt_records,
                pred_records=pred_rows,
                iou_thr=float(self.stage_eval_cfg.get("iou_thr", 0.5)),
                pck_thr=float(self.stage_eval_cfg.get("pck_thr", 0.2)),
            )
            dump_json(str(artifact_dir / "metrics.json"), metrics)
            metrics.pop("per_image", None)
            for key, value in _scalar_metrics(metrics).items():
                log_payload[f"stage_eval/{key}"] = value
            log_payload["stage_eval/num_samples"] = float(len(indices))

        if log_payload:
            swanlab.log(log_payload, step=int(getattr(state, "global_step", 0)))

    def on_train_begin(self, args, state, control, **kwargs):
        if not self._is_main_process(state):
            return
        self._ensure_run()
        swanlab = self._get_swanlab()
        swanlab.log(
            {
                "meta/train_dataset_size": float(len(self.train_dataset)),
                "meta/eval_dataset_size": float(len(self.eval_dataset)) if self.eval_dataset is not None else 0.0,
            },
            step=int(getattr(state, "global_step", 0)),
        )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not self._is_main_process(state):
            return
        metrics = _scalar_metrics(logs)
        if not metrics:
            return
        self._ensure_run()
        self._get_swanlab().log(metrics, step=int(getattr(state, "global_step", 0)))

    def on_evaluate(self, args, state, control, metrics=None, model=None, **kwargs):
        if not self._is_main_process(state) or model is None:
            return
        self._ensure_run()
        self._run_stage_eval(state, model)


def _prepare_preview_inputs(processor, prompt: str, image, max_length: int) -> Dict[str, torch.Tensor]:
    batch = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    return batch


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _filter_generate_inputs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    accepted = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "past_key_values",
        "use_cache",
    }
    for obj in _iter_signature_sources(model):
        for fn_name in ("forward", "prepare_inputs_for_generation"):
            fn = getattr(obj, fn_name, None)
            if fn is None:
                continue
            try:
                accepted.update(inspect.signature(fn).parameters.keys())
            except Exception:
                continue
    return {k: v for k, v in inputs.items() if k in accepted}


def _log_swanlab_predictions(
    trainer: Trainer,
    processor,
    dataset: CocoUniHumanSFTDataset,
    cfg: Dict[str, Any],
) -> None:
    swanlab_cfg = cfg.get("swanlab", {})
    if not isinstance(swanlab_cfg, dict) or not bool(swanlab_cfg.get("enable", False)):
        return

    try:
        import swanlab
    except Exception as e:
        raise RuntimeError("SwanLab preview logging requested, but `swanlab` is not installed.") from e

    preview_count = min(int(swanlab_cfg.get("preview_samples", 4)), len(dataset))
    if preview_count <= 0:
        return

    model = trainer.model
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else eos_token_id
    device = next(model.parameters()).device
    images_to_log = []

    with torch.no_grad():
        for idx in range(preview_count):
            sample = dataset[idx]
            prompt = build_vl_chat_text(
                processor=processor,
                prompt=str(sample["prompt"]),
                assistant_prefix=DEFAULT_ASSISTANT_PREFIX_TAG,
            )
            image = sample["image"]
            inputs = _prepare_preview_inputs(
                processor=processor,
                prompt=prompt,
                image=image,
                max_length=int(cfg.get("max_length", 4096)),
            )
            inputs = _move_batch_to_device(inputs, device)
            gen_inputs = _filter_generate_inputs(model, inputs)
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=int(swanlab_cfg.get("preview_max_new_tokens", 1024)),
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            input_len = gen_inputs["input_ids"].shape[1]
            pred_text = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
            caption = (
                f"id={sample['id']}\n"
                f"prompt={sample['prompt']}\n\n"
                f"gt={sample['target_text']}\n\n"
                f"pred={DEFAULT_ASSISTANT_PREFIX_TAG}{pred_text}"
            )
            images_to_log.append(swanlab.Image(image, caption=caption))

    if images_to_log:
        swanlab.log({"Prediction": images_to_log})


def _finish_swanlab(cfg: Dict[str, Any]) -> None:
    swanlab_cfg = cfg.get("swanlab", {})
    if not isinstance(swanlab_cfg, dict) or not bool(swanlab_cfg.get("enable", False)):
        return
    try:
        import swanlab
    except Exception:
        return
    swanlab.finish()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    set_seed(int(cfg.get("seed", 42)))

    model_name_or_path = str(cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    train_dataset = CocoUniHumanSFTDataset(
        jsonl_path=str(cfg["train_jsonl"]),
        image_root=cfg.get("image_root", None),
        max_samples=cfg.get("max_train_samples", None),
        extra_prompt=cfg.get("extra_prompt", None),
    )
    val_dataset = None
    if cfg.get("val_jsonl"):
        val_dataset = CocoUniHumanSFTDataset(
            jsonl_path=str(cfg["val_jsonl"]),
            image_root=cfg.get("image_root", None),
            max_samples=cfg.get("max_val_samples", None),
            extra_prompt=cfg.get("extra_prompt", None),
        )

    model = build_model(model_name_or_path=model_name_or_path, torch_dtype=get_dtype(cfg))
    if not _supports_vision_training(model, processor):
        raise ValueError(
            "This training pipeline expects a vision-language model with image inputs, "
            f"but `model_name_or_path={model_name_or_path}` does not expose a vision tower. "
            "Use a VL checkpoint such as Qwen2.5-VL instead of a text-only Qwen model."
        )
    model = maybe_apply_lora(model, cfg)
    if not _model_accepts_vision_inputs(model):
        raise ValueError(
            "Loaded model/adapter stack does not expose vision generate inputs "
            "(`pixel_values` / `image_grid_thw`) after wrapping. "
            f"Current model_name_or_path={model_name_or_path}."
        )
    if bool(cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    collator = SFTCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 4096)),
    )

    report_to = cfg.get("report_to", "none")
    if report_to in ("none", None):
        report_to = []

    callbacks: List[TrainerCallback] = [JsonlMetricCallback(output_dir=str(cfg["output_dir"]))]
    swanlab_cfg = cfg.get("swanlab", {})
    if isinstance(swanlab_cfg, dict) and bool(swanlab_cfg.get("enable", False)):
        callbacks.append(
            SwanLabVisualCallback(
                cfg=cfg,
                processor=processor,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
            )
        )

    has_eval = val_dataset is not None
    train_args = _build_training_arguments(cfg={**cfg, "report_to": report_to}, has_eval=has_eval)

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()
    if has_eval:
        metrics = trainer.evaluate()
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))
    preview_dataset = val_dataset if val_dataset is not None else train_dataset
    _log_swanlab_predictions(
        trainer=trainer,
        processor=processor,
        dataset=preview_dataset,
        cfg=cfg,
    )
    _finish_swanlab(cfg)


if __name__ == "__main__":
    main()
