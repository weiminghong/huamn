from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

from human_lisa_qwen_v1.inference_v1 import suppress_near_duplicate_person_records, summarize_inference_audit
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone
from human_lisa_qwen_v1.official_eval_v1 import encode_binary_mask
from human_lisa_qwen_v2.collator_v2 import _find_last_subsequence
from .modeling_stage1_v3 import Stage1V3Model
from human_lisa_qwen_v3_repr.schema_repr import (
    GUIDED_JSON_SCHEMA_V2,
    PROMPT_V2,
    canonicalize_person_ids_v2,
    validate_prediction_object_v2,
)
from .train_stage1_v3 import add_and_initialize_tokens, text_hidden_size


def build_chat_prefix(processor: Any) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_V2}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def make_guided_fn(tokenizer: Any, prompt_width: int):
    # Keep checkpoint loading and teacher-forced dense evaluation independent
    # from the optional constrained-generation backend.
    # lm-format-enforcer 0.10.x imports this class from the Transformers 4.x
    # location; Transformers 5 moved it to tokenization_utils_base.
    import transformers.tokenization_utils as tokenization_utils
    if not hasattr(tokenization_utils, "PreTrainedTokenizerBase"):
        from transformers.tokenization_utils_base import PreTrainedTokenizerBase
        tokenization_utils.PreTrainedTokenizerBase = PreTrainedTokenizerBase
    from lmformatenforcer import JsonSchemaParser
    from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

    base = getattr(tokenizer, "_human_v2_guided_json_base", None)
    if base is None:
        base = build_transformers_prefix_allowed_tokens_fn(tokenizer, JsonSchemaParser(GUIDED_JSON_SCHEMA_V2))
        tokenizer._human_v2_guided_json_base = base

    def allowed(batch_id: int, sequence: torch.Tensor):
        return base(batch_id, sequence[prompt_width:])

    return allowed


def decode_generated_json(tokenizer: Any, generated_ids: torch.Tensor) -> Tuple[str, bool]:
    ids = [int(value) for value in generated_ids.tolist()]
    eos_ids = {
        int(value)
        for value in [getattr(tokenizer, "eos_token_id", None), tokenizer.convert_tokens_to_ids("<|im_end|>")]
        if value is not None and int(value) >= 0
    }
    eos_seen = any(value in eos_ids for value in ids)
    text = tokenizer.decode(generated_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    for token in [getattr(tokenizer, "eos_token", None), "<|im_end|>", "<|endoftext|>"]:
        if token and token in text:
            text = text.split(token, 1)[0]
    return text.strip(), eos_seen


def load_stage1_v3_checkpoint(
    checkpoint_dir: str,
    *,
    base_model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> Tuple[Stage1V3Model, Any, Dict[str, Any]]:
    checkpoint = Path(checkpoint_dir)
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    base_model_path = base_model_path or manifest["model"]
    processor = AutoProcessor.from_pretrained(checkpoint / "processor", trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    base = load_qwen_vl_backbone(base_model_path, dtype)
    token_ids = add_and_initialize_tokens(processor, base)
    if token_ids != {name: int(value) for name, value in manifest["token_ids"].items()}:
        raise RuntimeError("processor/checkpoint special-token ids differ")
    from peft import PeftModel

    base = PeftModel.from_pretrained(base, checkpoint / "base_or_adapter", is_trainable=False)
    state = torch.load(checkpoint / "stage1_v3_task_modules.pt", map_location="cpu", weights_only=False)
    wrapper = Stage1V3Model(
        base_model=base,
        text_hidden_size=text_hidden_size(base),
        token_ids=token_ids,
        vision_layer_indices=state["vision_layer_indices"],
        decoder_dim=int(state["decoder_dim"]),
        decoder_heads=int(state["decoder_heads"]),
        person_query_layers=int(state["person_query_layers"]),
        pose_decoder_layers=int(state["pose_decoder_layers"]),
        visual_merge_size=int(state["visual_merge_size"]),
        max_people=int(state["max_people"]),
        loss_weights=state["loss_weights"],
    )
    wrapper.load_task_modules(str(checkpoint / "stage1_v3_task_modules.pt"))
    wrapper.to(device or torch.device("cuda")).eval()
    return wrapper, processor, manifest


def boxes_from_object(obj: Dict[str, Any], *, device: torch.device) -> torch.Tensor:
    boxes = torch.tensor([person["bbox"] for person in obj["persons"]], device=device, dtype=torch.float32).reshape(-1, 4)
    return boxes / 1000.0


def semantic_health_v3(
    obj: Dict[str, Any], *, boxes: torch.Tensor, dense: Dict[str, torch.Tensor], eos_seen: bool, raw_token_count: int
) -> Dict[str, Any]:
    errors: List[str] = []
    people = len(obj["persons"])
    if boxes.shape != (people, 4): errors.append("box_count_mismatch")
    if dense["pose_xy"].shape != (people, 17, 2): errors.append("pose_count_mismatch")
    if dense["mask_logits"].shape[0] != people: errors.append("seg_count_mismatch")
    if dense["person_probability"].shape[0] != people: errors.append("person_probability_count_mismatch")
    if dense["query_boxes_xyxy"].shape != (people, 4): errors.append("query_box_count_mismatch")
    if dense["query_objectness"].shape != (people,): errors.append("query_objectness_count_mismatch")
    if not eos_seen: errors.append("missing_eos")
    if raw_token_count >= 5000: errors.append("generation_hit_token_cap")
    if not torch.isfinite(boxes).all(): errors.append("nonfinite_boxes")
    if not torch.isfinite(dense["query_boxes_xyxy"]).all(): errors.append("nonfinite_query_boxes")
    if not torch.isfinite(dense["query_objectness"]).all(): errors.append("nonfinite_query_objectness")
    if boxes.numel() and ((boxes[:, 2] <= boxes[:, 0]).any() or (boxes[:, 3] <= boxes[:, 1]).any()):
        errors.append("invalid_box_geometry")
    duplicates = 0
    for first in range(people):
        for second in range(first + 1, people):
            a, b = boxes[first], boxes[second]
            tl = torch.maximum(a[:2], b[:2]); br = torch.minimum(a[2:], b[2:])
            intersection = (br - tl).clamp(min=0).prod()
            area_a = (a[2:] - a[:2]).clamp(min=0).prod(); area_b = (b[2:] - b[:2]).clamp(min=0).prod()
            if float(intersection / (area_a + area_b - intersection).clamp(min=1e-7)) >= 0.9: duplicates += 1
    if duplicates: errors.append("duplicate_boxes")
    return {"ok": not errors, "errors": errors, "persons": people, "duplicate_pairs": duplicates, "eos_seen": eos_seen, "raw_token_count": raw_token_count}


def assemble_coco_records_v3(
    *,
    image_id: int,
    width: int,
    height: int,
    obj: Dict[str, Any],
    boxes: torch.Tensor,
    dense: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    people = len(obj["persons"])
    if boxes.shape != (people, 4) or dense["pose_xy"].shape[0] != people or dense["mask_logits"].shape[0] != people:
        raise RuntimeError("person record and dense output counts differ")
    records: List[Dict[str, Any]] = []
    for index, person in enumerate(obj["persons"]):
        box = boxes[index].detach().float().cpu()
        x1, y1, x2, y2 = float(box[0] * width), float(box[1] * height), float(box[2] * width), float(box[3] * height)
        xy = dense["pose_xy"][index].detach().float().cpu()
        visibility = dense["pose_visibility_logits"][index].argmax(dim=-1).detach().cpu()
        keypoints: List[float] = []
        for joint in range(17):
            keypoints.extend([float(xy[joint, 0] * width), float(xy[joint, 1] * height), float(visibility[joint])])
        mask = F.interpolate(
            dense["mask_logits"][index : index + 1, None].float(),
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
                "score": float(dense["query_objectness"][index].clamp(0.0, 1.0)),
                "constant_score": 1.0,
                "person_token_probability": float(dense["person_probability"][index]),
                "query_objectness": float(dense["query_objectness"][index]),
                "query_box_xyxy_normalized": dense["query_boxes_xyxy"][index].detach().float().cpu().tolist(),
                "gender": int(person["gender"]),
                "age": float(person["age"]),
                "keypoints": keypoints,
                "segmentation": encode_binary_mask(mask.sigmoid().ge(0.5).detach().cpu().numpy()),
            }
        )
    return records


@torch.no_grad()
def decode_dense_for_text(
    model: Stage1V3Model,
    processor: Any,
    *,
    image: Any,
    raw_text: str,
    obj: Dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    prefix = build_chat_prefix(processor)
    target_text = raw_text + (processor.tokenizer.eos_token or "")
    processor.tokenizer.padding_side = "right"
    inputs = processor(
        text=[prefix + target_text], images=[image], return_tensors="pt", padding=True, truncation=False
    ).to(device)
    target_ids = processor.tokenizer(target_text, add_special_tokens=False)["input_ids"]
    sequence = [int(value) for value in inputs["input_ids"][0].tolist()]
    target_start = _find_last_subsequence(sequence, target_ids)
    if target_start < 0:
        raise RuntimeError("cannot locate generated JSON in multimodal sequence")
    if int(inputs["attention_mask"].sum()) > 5000:
        raise RuntimeError("generated dense decode exceeds max_length=5000")
    base_kwargs = {
        key: value
        for key, value in inputs.items()
        if key not in {"input_ids", "attention_mask", "image_grid_thw"}
    }
    generated_boxes = boxes_from_object(obj, device=device)
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.bfloat16):
        dense = model.decode_dense_from_inputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            image_grid_thw=inputs["image_grid_thw"],
            target_start=target_start,
            person_count=len(obj["persons"]),
            generated_boxes_xyxy=generated_boxes,
            **base_kwargs,
        )
    dense["boxes"] = generated_boxes
    dense["raw_token_count"] = torch.tensor(len(target_ids), device=device)
    return dense


@torch.no_grad()
def infer_samples_v3(
    model: Stage1V3Model,
    processor: Any,
    samples: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 5000,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    device = device or next(model.parameters()).device
    prefix = build_chat_prefix(processor)
    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=[prefix] * len(samples),
        images=[sample["image"] for sample in samples],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    prompt_width = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.autocast("cuda", dtype=dtype, enabled=device.type == "cuda" and dtype == torch.bfloat16):
        generation = model.base_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            prefix_allowed_tokens_fn=make_guided_fn(processor.tokenizer, prompt_width),
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
    generation_elapsed = time.perf_counter() - started
    results: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        raw, eos_seen = decode_generated_json(processor.tokenizer, generation[index, prompt_width:])
        result: Dict[str, Any] = {
            "image_id": int(sample["id"]),
            "raw": raw,
            "eos_seen": eos_seen,
            "schema_ok": False,
            "raw_schema_ok": False,
            "person_id_repairs": 0,
            "records": [],
            "score_strategy": "matched_query_objectness_native_text_box",
            "generation_latency_seconds": generation_elapsed / len(samples),
        }
        try:
            obj = json.loads(raw)
            repairs = canonicalize_person_ids_v2(obj)
            validate_prediction_object_v2(obj)
            dense_started = time.perf_counter()
            dense = decode_dense_for_text(
                model,
                processor,
                image=sample["image"],
                raw_text=raw,
                obj=obj,
                dtype=dtype,
                device=device,
            )
            result["dense_latency_seconds"] = time.perf_counter() - dense_started
            boxes = dense.pop("boxes")
            raw_token_count = int(dense.pop("raw_token_count"))
            health = semantic_health_v3(
                obj,
                boxes=boxes,
                dense=dense,
                eos_seen=eos_seen,
                raw_token_count=raw_token_count,
            )
            raw_records = assemble_coco_records_v3(
                image_id=int(sample["id"]),
                width=int(sample["width"]),
                height=int(sample["height"]),
                obj=obj,
                boxes=boxes,
                dense=dense,
            )
            records, suppression = suppress_near_duplicate_person_records(raw_records, iou_threshold=0.90)
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
            result["health"] = {"ok": False, "errors": ["inference_exception"]}
        results.append(result)
    return results
