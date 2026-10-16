#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.inference_v1 import (
    infer_samples,
    load_stage1_checkpoint,
    summarize_inference_audit,
)
from human_lisa_qwen_v1.official_eval_v1 import evaluate_all_official


DEFAULT_ATTR_EVAL = (
    REPO_ROOT
    / "HumanQueryNet_mmdet3_port"
    / "datasets"
    / "api_wrappers"
    / "eval_attr.py"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--annotation",
        default="/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json",
    )
    parser.add_argument(
        "--image-root",
        default="/share2/program/weiminghong/coco/val2017",
    )
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument(
        "--resume-from",
        default="",
        help="Seed this run from valid records in an interrupted inference.jsonl.",
    )
    parser.add_argument("--attr-eval-source", default=str(DEFAULT_ATTR_EVAL))
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def batches(
    dataset,
    batch_size: int,
    *,
    indices: Optional[List[int]] = None,
):
    order = list(range(len(dataset))) if indices is None else list(indices)
    for start in range(0, len(order), batch_size):
        yield [dataset[index] for index in order[start : start + batch_size]]


def load_resume_results(
    path: Path,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Any]]:
    path = Path(path)
    results: Dict[int, Dict[str, Any]] = {}
    audit = {
        "path": str(path),
        "lines": 0,
        "valid": 0,
        "recovered": 0,
        "invalid": 0,
        "duplicates": 0,
    }
    if not path.is_file():
        return results, audit
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            audit["lines"] += 1
            line = raw_line.strip()
            if not line:
                audit["invalid"] += 1
                continue
            recovered = False
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                object_start = line.find("{")
                if object_start < 0:
                    audit["invalid"] += 1
                    continue
                try:
                    row = json.loads(line[object_start:])
                    recovered = True
                except json.JSONDecodeError:
                    audit["invalid"] += 1
                    continue
            try:
                image_id = int(row["image_id"])
            except (KeyError, TypeError, ValueError):
                audit["invalid"] += 1
                continue
            if image_id in results:
                audit["duplicates"] += 1
                continue
            results[image_id] = row
            audit["valid"] += 1
            audit["recovered"] += int(recovered)
    return results, audit


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("evaluation requires CUDA")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda")
    model, processor, manifest = load_stage1_checkpoint(
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
    inference_path = output_dir / "inference.jsonl"
    dataset_ids = [int(image["id"]) for image in dataset.images]
    dataset_id_set = set(dataset_ids)
    resume_results, resume_audit = load_resume_results(
        Path(args.resume_from) if args.resume_from else Path("")
    )
    results_by_image = {
        image_id: result
        for image_id, result in resume_results.items()
        if image_id in dataset_id_set
    }
    resume_audit["outside_dataset"] = len(resume_results) - len(results_by_image)
    pending_indices = [
        index
        for index, image_id in enumerate(dataset_ids)
        if image_id not in results_by_image
    ]
    schema_ok = sum(
        int(bool(result.get("schema_ok"))) for result in results_by_image.values()
    )
    health_ok = sum(
        int(bool((result.get("health") or {}).get("ok")))
        for result in results_by_image.values()
    )
    with inference_path.open("w", encoding="utf-8") as handle:
        for image_id in dataset_ids:
            if image_id in results_by_image:
                handle.write(
                    json.dumps(results_by_image[image_id], ensure_ascii=False) + "\n"
                )
        handle.flush()
        processed = len(results_by_image)
        print(
            json.dumps(
                {
                    "resume": resume_audit,
                    "processed": processed,
                    "total": len(dataset),
                    "remaining": len(pending_indices),
                }
            ),
            flush=True,
        )
        for sample_batch in batches(
            dataset, args.batch_size, indices=pending_indices
        ):
            results = infer_samples(
                model,
                processor,
                sample_batch,
                max_new_tokens=args.max_new_tokens,
                dtype=dtype,
                device=device,
            )
            for result in results:
                image_id = int(result["image_id"])
                results_by_image[image_id] = result
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                handle.flush()
                schema_ok += int(bool(result.get("schema_ok")))
                health_ok += int(bool((result.get("health") or {}).get("ok")))
                processed += 1
            print(
                json.dumps(
                    {
                        "processed": processed,
                        "total": len(dataset),
                        "schema_ok": schema_ok,
                        "health_ok": health_ok,
                    }
                ),
                flush=True,
            )

    missing_ids = [
        image_id for image_id in dataset_ids if image_id not in results_by_image
    ]
    if missing_ids:
        raise RuntimeError(
            f"evaluation ended with {len(missing_ids)} missing images; "
            f"first missing ids: {missing_ids[:10]}"
        )
    all_results = [results_by_image[image_id] for image_id in dataset_ids]
    all_records = [
        record
        for result in all_results
        for record in result.get("records", [])
    ]
    image_ids = list(dataset_ids)

    official = evaluate_all_official(
        args.annotation,
        all_records,
        image_ids=image_ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    count_errors = []
    for result, image_info in zip(all_results, dataset.images):
        gt_count = sum(
            not bool(annotation.get("iscrowd", 0))
            and float(annotation.get("area", 0)) > 0
            for annotation in dataset.annotations_by_image.get(
                int(image_info["id"]), []
            )
        )
        count_errors.append(abs(len(result.get("records", [])) - gt_count))
    health = {
        **summarize_inference_audit(all_results),
        "person_count_mae": sum(count_errors) / max(1, len(count_errors)),
        "predicted_persons": len(all_records),
        "inference_file": str(inference_path),
        "resume_audit": resume_audit,
    }
    metrics = {
        **official,
        **health,
        "checkpoint": str(args.checkpoint),
        "annotation": str(args.annotation),
        "protocol": "official_cocoeval_and_humanquerynet_attr",
        "manifest": manifest,
    }
    with (output_dir / "predictions.coco_multitask.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(all_records, handle, ensure_ascii=False)
    with (output_dir / "metrics.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

