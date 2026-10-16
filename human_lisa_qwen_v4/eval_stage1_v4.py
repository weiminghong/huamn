#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.official_eval_v1 import evaluate_all_official

from .inference_v4 import (
    infer_samples_v4,
    load_stage1_v4_checkpoint,
    summarize_inference_audit_v4,
)
from .matching_v4 import hungarian_box_match


REPO_ROOT = Path(__file__).resolve().parents[1]
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
        default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json",
    )
    parser.add_argument(
        "--image-root", default="/root/autodl-tmp/coco/val2017"
    )
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=5000)
    parser.add_argument("--resume-from", default="")
    parser.add_argument("--attr-eval-source", default=str(DEFAULT_ATTR_EVAL))
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def batches(dataset: Any, batch_size: int, indices: List[int]):
    for start in range(0, len(indices), batch_size):
        yield [dataset[index] for index in indices[start : start + batch_size]]


def load_resume(path: Path) -> Dict[int, Dict[str, Any]]:
    if not path.is_file():
        return {}
    result: Dict[int, Dict[str, Any]] = {}
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
            result[int(value["image_id"])] = value
        except Exception as error:
            raise RuntimeError(
                f"invalid resume line {line_number}: {error}"
            ) from error
    return result


def _valid_gt_boxes(dataset: Any, image_id: int) -> List[List[float]]:
    boxes = []
    for annotation in dataset.annotations_by_image.get(int(image_id), []):
        valid = annotation.get("valid") or {}
        bbox = annotation.get("bbox") or []
        if (
            bool(annotation.get("iscrowd", 0))
            or bool(annotation.get("ignore", 0))
            or len(bbox) != 4
            or float(bbox[2]) <= 0
            or float(bbox[3]) <= 0
            or float(annotation.get("area", bbox[2] * bbox[3])) <= 0
            or ("bboxes" in valid and not bool(int(valid["bboxes"])))
        ):
            continue
        x, y, width, height = [float(value) for value in bbox]
        boxes.append([x, y, x + width, y + height])
    return boxes


def _predicted_boxes(result: Dict[str, Any]) -> List[List[float]]:
    boxes = []
    for record in result.get("records", []):
        x, y, width, height = [float(value) for value in record["bbox"]]
        boxes.append([x, y, x + width, y + height])
    return boxes


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_new_tokens != 5000:
        raise ValueError(
            "v4 evaluation requires batch_size>0 and max_new_tokens=5000"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("v4 generated evaluation requires CUDA")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda")
    model, processor, manifest = load_stage1_v4_checkpoint(
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
    results = {
        image_id: value
        for image_id, value in resumed.items()
        if image_id in id_set
    }
    pending = [
        index
        for index, image_id in enumerate(ids)
        if image_id not in results
    ]
    inference_path = output / "inference.jsonl"
    status_path = output / "status.json"
    with inference_path.open("w", encoding="utf-8") as handle:
        for image_id in ids:
            if image_id in results:
                handle.write(
                    json.dumps(results[image_id], ensure_ascii=False) + "\n"
                )
        handle.flush()
        for sample_batch in batches(dataset, args.batch_size, pending):
            values = infer_samples_v4(
                model,
                processor,
                sample_batch,
                max_new_tokens=args.max_new_tokens,
                dtype=dtype,
                device=device,
            )
            for value in values:
                image_id = int(value["image_id"])
                results[image_id] = value
                handle.write(json.dumps(value, ensure_ascii=False) + "\n")
                handle.flush()
            audit = summarize_inference_audit_v4(list(results.values()))
            status = {
                "processed": len(results),
                "total": len(dataset),
                "schema_ok": audit["schema_ok_rate"],
                "health_ok": audit["health_ok_rate"],
                "checkpoint": args.checkpoint,
            }
            status_path.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            print(json.dumps(status), flush=True)
    missing = [image_id for image_id in ids if image_id not in results]
    if missing:
        raise RuntimeError(f"evaluation ended with {len(missing)} missing images")
    ordered = [results[image_id] for image_id in ids]
    records = [
        record for result in ordered for record in result.get("records", [])
    ]
    official = evaluate_all_official(
        args.annotation,
        records,
        image_ids=ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    constant = evaluate_all_official(
        args.annotation,
        [{**record, "score": 1.0} for record in records],
        image_ids=ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    gt_persons = 0
    count_errors = []
    matched_iou50 = 0
    for result, image_id in zip(ordered, ids):
        gt_boxes = _valid_gt_boxes(dataset, image_id)
        pred_boxes = _predicted_boxes(result)
        gt_persons += len(gt_boxes)
        count_errors.append(abs(len(pred_boxes) - len(gt_boxes)))
        matched_iou50 += len(
            hungarian_box_match(pred_boxes, gt_boxes, minimum_iou=0.5)
        )
    predicted_persons = len(records)
    metrics = {
        **official,
        **{f"constant_score_{name}": value for name, value in constant.items()},
        **summarize_inference_audit_v4(ordered),
        "pose_AP": official.get("keypoints_AP", 0.0),
        "pose_AP50": official.get("keypoints_AP50", 0.0),
        "pose_AP75": official.get("keypoints_AP75", 0.0),
        "person_count_mae": sum(count_errors) / max(1, len(count_errors)),
        "gt_persons": gt_persons,
        "predicted_persons": predicted_persons,
        "matched_persons_iou50": matched_iou50,
        "box_recall_iou50": matched_iou50 / max(1, gt_persons),
        "box_precision_iou50": matched_iou50 / max(1, predicted_persons),
        "checkpoint": args.checkpoint,
        "annotation": args.annotation,
        "split": (
            "full_val" if args.max_samples is None else "diagnostic_val_subset"
        ),
        "score_strategy": "generated_person_record_mean_logprob",
        "protocol": (
            "stage1_v4_guided_json_native_text_box_record_bound_pose_seg_"
            "official_cocoeval_partial_label_attr"
        ),
        "manifest": manifest,
    }
    (output / "predictions.coco_multitask.json").write_text(
        json.dumps(records, ensure_ascii=False), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
