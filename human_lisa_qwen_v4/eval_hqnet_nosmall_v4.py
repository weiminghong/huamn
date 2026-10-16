#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pycocotools
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from human_lisa_qwen_v1.official_eval_v1 import (
    _load_attr_evaluator,
    _partial_label_attr_evaluator,
)

AREA_MIN = 32.0 ** 2
AREA_MEDIUM_MAX = 96.0 ** 2
AREA_MAX = 1e5 ** 2

ANNOTATION = Path("/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json")
SOURCE_DIR = Path(
    "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/"
    "full64115_2ep_bs8ga2_20260805/"
    "eval_epoch1_generated_full_val2693_bs1_tok5000_20260807"
)
PREDICTIONS = SOURCE_DIR / "predictions.coco_multitask.json"
SOURCE_METRICS = SOURCE_DIR / "metrics.json"
ATTR_SOURCE = Path(
    "/root/autodl-tmp/humanslot/HumanQueryNet_mmdet3_port/"
    "datasets/api_wrappers/eval_attr.py"
)
OUTPUT = Path(
    "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/"
    "full64115_2ep_bs8ga2_20260805/"
    "eval_epoch1_generated_full_val2693_hqnet_nosmall_20260809"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_no_small_ranges(evaluator: Any) -> None:
    evaluator.params.areaRng = [
        [AREA_MIN, AREA_MAX],
        [AREA_MIN, AREA_MEDIUM_MAX],
        [AREA_MEDIUM_MAX, AREA_MAX],
    ]
    evaluator.params.areaRngLbl = ["all", "medium", "large"]


def mean_valid(values: np.ndarray) -> float:
    values = np.asarray(values)
    values = values[values > -1]
    return float(values.mean()) if values.size else -1.0


def summarize_ap(evaluator: Any, area_index: int, iou: float | None = None) -> float:
    precision = evaluator.eval["precision"]
    if iou is not None:
        threshold_index = np.where(np.isclose(evaluator.params.iouThrs, iou))[0]
        if threshold_index.size != 1:
            raise RuntimeError(f"cannot locate IoU threshold {iou}")
        precision = precision[threshold_index]
    precision = precision[:, :, :, area_index, -1]
    return mean_valid(precision)


def run_coco(
    coco_gt: COCO,
    records: List[Dict[str, Any]],
    iou_type: str,
    image_ids: Sequence[int],
) -> Dict[str, float]:
    coco_dt = coco_gt.loadRes(records)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = [int(value) for value in image_ids]
    set_no_small_ranges(evaluator)
    evaluator.evaluate()
    evaluator.accumulate()
    return {
        "AP": summarize_ap(evaluator, 0),
        "AP50": summarize_ap(evaluator, 0, 0.50),
        "AP75": summarize_ap(evaluator, 0, 0.75),
        "APm": summarize_ap(evaluator, 1),
        "APl": summarize_ap(evaluator, 2),
    }


def prediction_attr(record: Dict[str, Any], name: str) -> float:
    if name != "gender":
        return float(record.get(name, -1))
    if "gender_score" in record:
        return float(record["gender_score"])
    value = int(record.get("gender", -1))
    return 1.0 if value == 0 else (0.0 if value == 1 else -1.0)


def run_attr(
    coco_gt: COCO,
    predictions: List[Dict[str, Any]],
    name: str,
    image_ids: Sequence[int],
) -> Dict[str, float]:
    records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "bbox": [float(value) for value in record["bbox"]],
            "score": float(record["score"]),
            name: prediction_attr(record, name),
        }
        for record in predictions
        if "bbox" in record
    ]
    evaluator_cls = _partial_label_attr_evaluator(_load_attr_evaluator(ATTR_SOURCE))
    evaluator = evaluator_cls(
        coco_gt,
        coco_gt.loadRes(records),
        iouType="bbox",
        attr_type=name,
    )
    evaluator.params.imgIds = [int(value) for value in image_ids]
    set_no_small_ranges(evaluator)
    evaluator.evaluate()
    evaluator.accumulate()
    return {
        "AP": summarize_ap(evaluator, 0),
        "AP50": summarize_ap(evaluator, 0, 0.50),
        "AP75": summarize_ap(evaluator, 0, 0.75),
        "APm": summarize_ap(evaluator, 1),
        "APl": summarize_ap(evaluator, 2),
    }


def valid_non_small_gt(coco_gt: COCO, image_ids: Sequence[int]) -> Dict[str, int]:
    selected = set(int(value) for value in image_ids)
    counts = {
        "gt_persons_medium_large": 0,
        "gt_persons_medium": 0,
        "gt_persons_large": 0,
        "gender_gt_labeled_medium_large": 0,
        "age_gt_labeled_medium_large": 0,
    }
    for ann in coco_gt.dataset.get("annotations", []):
        if int(ann.get("image_id", -1)) not in selected:
            continue
        valid = ann.get("valid") or {}
        bbox = ann.get("bbox") or []
        if (
            bool(ann.get("iscrowd", 0))
            or bool(ann.get("ignore", 0))
            or len(bbox) != 4
            or float(bbox[2]) <= 0
            or float(bbox[3]) <= 0
            or ("bboxes" in valid and not bool(int(valid["bboxes"])))
        ):
            continue
        area = float(ann.get("area", float(bbox[2]) * float(bbox[3])))
        if area < AREA_MIN:
            continue
        counts["gt_persons_medium_large"] += 1
        if area < AREA_MEDIUM_MAX:
            counts["gt_persons_medium"] += 1
        else:
            counts["gt_persons_large"] += 1
        for name in ("gender", "age"):
            if float(ann.get(name, -1)) > -1 and (
                name not in valid or bool(int(valid[name]))
            ):
                counts[f"{name}_gt_labeled_medium_large"] += 1
    return counts


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions = json.loads(PREDICTIONS.read_text(encoding="utf-8"))
    source_metrics = json.loads(SOURCE_METRICS.read_text(encoding="utf-8"))
    coco_gt = COCO(str(ANNOTATION))
    image_ids = sorted(int(value) for value in coco_gt.getImgIds())

    bbox_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "bbox": [float(value) for value in record["bbox"]],
            "score": float(record["score"]),
        }
        for record in predictions
        if "bbox" in record
    ]
    keypoint_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "keypoints": [float(value) for value in record["keypoints"]],
            "score": float(record["score"]),
        }
        for record in predictions
        if len(record.get("keypoints", [])) == 51
    ]
    segm_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "segmentation": record["segmentation"],
            "score": float(record["score"]),
        }
        for record in predictions
        if "segmentation" in record
    ]

    groups = {
        "bbox": run_coco(coco_gt, bbox_records, "bbox", image_ids),
        "keypoints": run_coco(coco_gt, keypoint_records, "keypoints", image_ids),
        "segm": run_coco(coco_gt, segm_records, "segm", image_ids),
        "gender": run_attr(coco_gt, predictions, "gender", image_ids),
        "age": run_attr(coco_gt, predictions, "age", image_ids),
    }
    metrics = {
        f"{group}_{name}": value
        for group, values in groups.items()
        for name, value in values.items()
    }
    metrics.update(valid_non_small_gt(coco_gt, image_ids))
    metrics.update(
        {
            "images": len(image_ids),
            "predicted_persons": len(predictions),
            "checkpoint": source_metrics.get("checkpoint"),
            "source_prediction_dir": str(SOURCE_DIR),
            "annotation": str(ANNOTATION),
            "split": "full_val",
            "area_policy": {
                "all": [AREA_MIN, AREA_MAX],
                "medium": [AREA_MIN, AREA_MEDIUM_MAX],
                "large": [AREA_MEDIUM_MAX, AREA_MAX],
                "boundary_semantics": "COCO area-range ignore rules",
            },
            "protocol": (
                "stage1_v4_generated_existing_predictions_"
                "hqnet_paper_no_small_area_aligned_cocoeval"
            ),
            "result_classification": (
                "paper-aligned-rescore; no new inference; "
                "official-protocol candidate pending official checkpoint parity"
            ),
        }
    )

    checks = {}
    for group in ("bbox", "keypoints", "segm"):
        for suffix in ("APm", "APl"):
            original_key = f"{group}_{suffix}"
            aligned_key = original_key
            original = float(source_metrics[original_key])
            aligned = float(metrics[aligned_key])
            checks[f"{group}_{suffix}_matches_standard_slice"] = {
                "original": original,
                "aligned": aligned,
                "absolute_difference": abs(original - aligned),
                "pass": abs(original - aligned) < 1e-12,
            }
    if not all(value["pass"] for value in checks.values()):
        raise RuntimeError(f"area-slice validation failed: {checks}")

    manifest = {
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        "annotation": str(ANNOTATION),
        "annotation_sha256": sha256(ANNOTATION),
        "predictions": str(PREDICTIONS),
        "predictions_sha256": sha256(PREDICTIONS),
        "source_metrics": str(SOURCE_METRICS),
        "source_metrics_sha256": sha256(SOURCE_METRICS),
        "images": len(image_ids),
        "predictions_count": len(predictions),
        "seed": None,
        "deterministic": True,
        "new_inference": False,
        "metric_definition": (
            "COCO AP with overall area range [32^2, 1e5^2], "
            "plus standard medium and large ranges"
        ),
        "aggregation": "mean valid precision over IoU, recall, and category",
        "classification": metrics["result_classification"],
        "validation_checks": checks,
    }
    (OUTPUT / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (OUTPUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"metrics": metrics, "validation_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
