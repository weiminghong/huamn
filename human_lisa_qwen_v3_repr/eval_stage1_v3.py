#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.official_eval_v1 import evaluate_all_official
from human_lisa_qwen_v3_repr.inference_v3 import infer_samples_v3, load_stage1_v3_checkpoint, summarize_inference_audit

DEFAULT_ATTR_EVAL = REPO_ROOT / "HumanQueryNet_mmdet3_port" / "datasets" / "api_wrappers" / "eval_attr.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json")
    parser.add_argument("--image-root", default="/root/autodl-tmp/coco/val2017")
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--attr-eval-source", default=str(DEFAULT_ATTR_EVAL))
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def batches(dataset: Any, batch_size: int, indices: List[int]):
    for start in range(0, len(indices), batch_size):
        yield [dataset[index] for index in indices[start : start + batch_size]]


def load_resume(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            result[int(value["image_id"])] = value
        except Exception as error:
            raise RuntimeError(f"invalid resume line {line_number}: {error}") from error
    return result


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens != 5000:
        raise ValueError("V3 evaluation requires batch_size>0 and max_new_tokens=5000")
    if not torch.cuda.is_available():
        raise RuntimeError("V3 evaluation requires CUDA")
    output_dir = Path(args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda")
    model, processor, manifest = load_stage1_v3_checkpoint(
        args.checkpoint,
        base_model_path=args.base_model or None,
        dtype=dtype,
        device=device,
    )
    dataset = CocoUniHumanV1Dataset(
        args.annotation,
        args.image_root,
        training=False,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples,
        decode_eval_segmentation=False,
    )
    ids = [int(image["id"]) for image in dataset.images]
    id_set = set(ids)
    resumed = load_resume(Path(args.resume_from)) if args.resume_from else {}
    results = {image_id: value for image_id, value in resumed.items() if image_id in id_set}
    pending = [index for index, image_id in enumerate(ids) if image_id not in results]
    inference_path = output_dir / "inference.jsonl"
    status_path = output_dir / "status.json"
    schema_ok = sum(bool(value.get("schema_ok")) for value in results.values())
    health_ok = sum(bool((value.get("health") or {}).get("ok")) for value in results.values())
    with inference_path.open("w", encoding="utf-8") as handle:
        for image_id in ids:
            if image_id in results:
                handle.write(json.dumps(results[image_id], ensure_ascii=False) + "\n")
        handle.flush()
        for sample_batch in batches(dataset, args.batch_size, pending):
            for value in infer_samples_v3(
                model,
                processor,
                sample_batch,
                max_new_tokens=args.max_new_tokens,
                dtype=dtype,
                device=device,
            ):
                image_id = int(value["image_id"])
                results[image_id] = value
                handle.write(json.dumps(value, ensure_ascii=False) + "\n"); handle.flush()
                schema_ok += int(bool(value.get("schema_ok")))
                health_ok += int(bool((value.get("health") or {}).get("ok")))
            status = {
                "processed": len(results),
                "total": len(dataset),
                "schema_ok": schema_ok,
                "health_ok": health_ok,
                "checkpoint": args.checkpoint,
            }
            status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status), flush=True)
    missing = [image_id for image_id in ids if image_id not in results]
    if missing:
        raise RuntimeError(f"evaluation ended with {len(missing)} missing images")
    ordered = [results[image_id] for image_id in ids]
    records = [record for result in ordered for record in result.get("records", [])]
    official = evaluate_all_official(
        args.annotation,
        records,
        image_ids=ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    constant_score_official = evaluate_all_official(
        args.annotation,
        [{**record, "score": 1.0} for record in records],
        image_ids=ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    count_errors = []
    gt_people = 0
    for result, image_info in zip(ordered, dataset.images):
        count = sum(
            not bool(annotation.get("iscrowd", 0)) and float(annotation.get("area", 0)) > 0
            for annotation in dataset.annotations_by_image.get(int(image_info["id"]), [])
        )
        gt_people += count
        count_errors.append(abs(len(result.get("records", [])) - count))
    metrics = {
        **official,
        **{f"constant_score_{name}": value for name, value in constant_score_official.items()},
        **summarize_inference_audit(ordered),
        "person_count_mae": sum(count_errors) / max(1, len(count_errors)),
        "gt_persons": gt_people,
        "predicted_persons": len(records),
        "checkpoint": args.checkpoint,
        "annotation": args.annotation,
        "split": "full_val" if args.max_samples is None else "diagnostic_val_subset",
        "score_strategy": "matched_query_objectness_native_text_box; constant-score metrics are retained as a diagnostic",
        "protocol": "stage1_v3_official_cocoeval_partial_label_attr_native_text_box_query_confidence_query_bound_dense_tasks",
        "manifest": manifest,
    }
    (output_dir / "predictions.coco_multitask.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
