#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
from transformers import AutoProcessor

try:
    import transformers.tokenization_utils as _tokenization_utils
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase as _PreTrainedTokenizerBase
    if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
        _tokenization_utils.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
except Exception:
    pass

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone
from human_lisa_qwen_v1.official_eval_v1 import _run_coco_eval, evaluate_all_official


PROMPT = """Detect every clear visible real person in the image exactly once.
Include clear partially visible people. Skip reflections, posters, statues, mannequins, isolated body parts, and ambiguous tiny regions.
Order persons left-to-right by bounding-box center, breaking ties top-to-bottom.
Return JSON only. Do not explain or think aloud.
For each person:
- person_id: consecutive integers starting at 1.
- bbox: [x1,y1,x2,y2] as tight integer coordinates normalized to [0,1000]. Require x1<x2 and y1<y2.
- gender: 0 male, 1 female, -1 unknown.
- age: estimated age in years, or -1 unknown.
- score: estimated detection confidence as an integer from 0 to 1000.
Stop immediately after the final JSON object. Do not invent extra persons to fill the response."""

SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "persons": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "person_id": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "gender": {"type": "integer", "enum": [-1, 0, 1]},
                    "age": {"type": "integer", "minimum": -1, "maximum": 120},
                    "score": {"type": "integer", "minimum": 0, "maximum": 1000},
                },
                "required": ["person_id", "bbox", "gender", "age", "score"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["persons"],
    "additionalProperties": False,
}

DEFAULT_ATTR_EVAL = Path("/root/autodl-tmp/humanslot/HumanQueryNet_mmdet3_port/datasets/api_wrappers/eval_attr.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def build_prefix(processor: Any) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def guided_fn(tokenizer: Any, prompt_width: int):
    parser = JsonSchemaParser(SCHEMA)
    base_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    def allowed(batch_id: int, sent: torch.Tensor):
        return base_fn(batch_id, sent[prompt_width:])

    return allowed


def decode_json(tokenizer: Any, token_ids: torch.Tensor) -> tuple[str, Any, bool]:
    ids = [int(value) for value in token_ids.tolist()]
    eos_ids = {
        int(value)
        for value in [getattr(tokenizer, "eos_token_id", None), tokenizer.convert_tokens_to_ids("<|im_end|>")]
        if value is not None and int(value) >= 0
    }
    eos_seen = any(value in eos_ids for value in ids)
    raw = tokenizer.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < start:
        return raw, None, eos_seen
    candidate = raw[start : end + 1]
    try:
        return candidate, json.loads(candidate), eos_seen
    except json.JSONDecodeError:
        return candidate, None, eos_seen


def validate_object(obj: Any) -> bool:
    if not isinstance(obj, dict) or set(obj) != {"persons"} or not isinstance(obj["persons"], list):
        return False
    for person in obj["persons"]:
        if not isinstance(person, dict) or set(person) != {"person_id", "bbox", "gender", "age", "score"}:
            return False
        bbox = person["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4 or not all(isinstance(v, int) and 0 <= v <= 1000 for v in bbox):
            return False
        if not isinstance(person["person_id"], int) or person["person_id"] < 1:
            return False
        if person["gender"] not in (-1, 0, 1):
            return False
        if not isinstance(person["age"], int) or not -1 <= person["age"] <= 120:
            return False
        if not isinstance(person["score"], int) or not 0 <= person["score"] <= 1000:
            return False
    return True


def box_iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in first]
    bx, by, bw, bh = [float(v) for v in second]
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1, ix2, iy2 = max(ax, bx), max(ay, by), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / max(1e-12, union)


def records_from_object(image_id: int, width: int, height: int, obj: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
    records: List[Dict[str, Any]] = []
    invalid = 0
    for index, person in enumerate(obj.get("persons", []), start=1):
        x1, y1, x2, y2 = [float(value) for value in person["bbox"]]
        x1, x2 = x1 * width / 1000.0, x2 * width / 1000.0
        y1, y2 = y1 * height / 1000.0, y2 * height / 1000.0
        x1, y1 = min(max(x1, 0.0), float(width)), min(max(y1, 0.0), float(height))
        x2, y2 = min(max(x2, 0.0), float(width)), min(max(y2, 0.0), float(height))
        if x2 <= x1 or y2 <= y1:
            invalid += 1
            continue
        records.append({
            "image_id": int(image_id),
            "category_id": 1,
            "person_id": int(person.get("person_id", index)),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": float(person["score"]) / 1000.0,
            "gender": int(person["gender"]),
            "age": float(person["age"]),
        })
    return records, invalid


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    inference_path = output_dir / "inference.jsonl"
    metrics_path = output_dir / "metrics.json"

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id
    model = load_qwen_vl_backbone(args.model, torch_dtype=dtype).to(device).eval()
    dataset = CocoUniHumanV1Dataset(
        args.annotation,
        args.image_root,
        training=False,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples,
        decode_eval_segmentation=False,
    )
    prefix = build_prefix(processor)
    loaded_by_id: Dict[int, Dict[str, Any]] = {}
    if args.resume and inference_path.exists():
        with inference_path.open("r", encoding="utf-8") as source:
            for line in source:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    loaded_by_id[int(row["image_id"])] = row
                except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                    continue
    ordered_ids = [int(image["id"]) for image in dataset.images]
    rows: List[Dict[str, Any]] = [loaded_by_id[image_id] for image_id in ordered_ids if image_id in loaded_by_id]
    resumed_rows = len(rows)
    all_records: List[Dict[str, Any]] = [record for row in rows for record in row.get("records", [])]
    image_ids: List[int] = [int(row["image_id"]) for row in rows]
    count_errors: List[int] = [abs(int(row.get("pred_persons", 0)) - int(row.get("gt_persons", 0))) for row in rows]
    token_counts: List[int] = [int(row.get("generated_tokens", 0)) for row in rows]
    pending_indices = [index for index, image in enumerate(dataset.images) if int(image["id"]) not in loaded_by_id]
    started = time.perf_counter()
    file_mode = "a" if rows else "w"
    with inference_path.open(file_mode, encoding="utf-8") as sink, torch.no_grad():
        for start in range(0, len(pending_indices), args.batch_size):
            indices = pending_indices[start : start + args.batch_size]
            samples = [dataset[index] for index in indices]
            inputs = processor(
                text=[prefix] * len(samples),
                images=[sample["image"] for sample in samples],
                return_tensors="pt",
                padding=True,
                truncation=False,
            ).to(device)
            prompt_width = int(inputs["input_ids"].shape[1])
            batch_started = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=args.bf16):
                generated = model.generate(
                    **inputs,
                    max_new_tokens=int(args.max_new_tokens),
                    do_sample=False,
                    num_beams=1,
                    prefix_allowed_tokens_fn=guided_fn(processor.tokenizer, prompt_width),
                    eos_token_id=processor.tokenizer.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    use_cache=True,
                )
            batch_seconds = time.perf_counter() - batch_started
            for row_index, sample in enumerate(samples):
                suffix = generated[row_index, prompt_width:]
                raw, obj, eos_seen = decode_json(processor.tokenizer, suffix)
                schema_ok = validate_object(obj)
                records, invalid_boxes = records_from_object(
                    int(sample["id"]), int(sample["width"]), int(sample["height"]), obj if schema_ok else {"persons": []}
                )
                duplicate_pairs = sum(
                    int(box_iou_xywh(records[i]["bbox"], records[j]["bbox"]) >= 0.9)
                    for i in range(len(records)) for j in range(i + 1, len(records))
                )
                gt_count = len(sample.get("instances", []))
                count_errors.append(abs(len(records) - gt_count))
                token_count = int(suffix.ne(processor.tokenizer.pad_token_id).sum().item())
                token_counts.append(token_count)
                row = {
                    "image_id": int(sample["id"]),
                    "raw": raw,
                    "schema_ok": bool(schema_ok),
                    "eos_seen": bool(eos_seen),
                    "records": records,
                    "gt_persons": gt_count,
                    "pred_persons": len(records),
                    "invalid_boxes": invalid_boxes,
                    "duplicate_pairs_iou90": duplicate_pairs,
                    "generated_tokens": token_count,
                    "generation_seconds": batch_seconds / max(1, len(samples)),
                }
                rows.append(row)
                all_records.extend(records)
                image_ids.append(int(sample["id"]))
                sink.write(json.dumps(row, ensure_ascii=False) + "\n")
                sink.flush()
                print(json.dumps({key: row[key] for key in ["image_id", "schema_ok", "gt_persons", "pred_persons", "generated_tokens", "generation_seconds"]}, ensure_ascii=False), flush=True)

    official = evaluate_all_official(
        args.annotation,
        all_records,
        image_ids=image_ids,
        attr_eval_source=DEFAULT_ATTR_EVAL,
    )
    constant_records = [{**record, "score": 1.0} for record in all_records]
    from pycocotools.coco import COCO
    constant_bbox = _run_coco_eval(COCO(args.annotation), constant_records, iou_type="bbox", image_ids=image_ids)
    metrics = {
        "protocol": "diagnostic_native_qwen35_zero_shot_guided_json_box_gender_age",
        "paper_eligibility": "diagnostic_until_score_protocol_is_fixed",
        "images": len(rows),
        "gt_persons": sum(row["gt_persons"] for row in rows),
        "pred_persons": len(all_records),
        "schema_ok_rate": sum(int(row["schema_ok"]) for row in rows) / max(1, len(rows)),
        "eos_rate": sum(int(row["eos_seen"]) for row in rows) / max(1, len(rows)),
        "person_count_mae": statistics.fmean(count_errors) if count_errors else 0.0,
        "duplicate_pairs_iou90": sum(row["duplicate_pairs_iou90"] for row in rows),
        "invalid_boxes": sum(row["invalid_boxes"] for row in rows),
        "avg_generated_tokens": statistics.fmean(token_counts) if token_counts else 0.0,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "generation_seconds_sum": sum(float(row.get("generation_seconds", 0.0)) for row in rows),
        "resumed_rows": resumed_rows,
        "batch_size": int(args.batch_size),
        "max_new_tokens": int(args.max_new_tokens),
        "score_protocol": "self_reported_integer_0_1000_for_primary; constant_1_for_sensitivity",
        **{key: value for key, value in official.items() if key.startswith(("bbox_", "gender_", "age_"))},
        **{f"bbox_constant_score_{key}": value for key, value in constant_bbox.items()},
        "model": args.model,
        "annotation": args.annotation,
        "prompt": PROMPT,
        "schema": SCHEMA,
    }
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
