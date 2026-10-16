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

sys.path.insert(0, "/root/autodl-tmp/humanslot")
from human_lisa_qwen_v1.official_eval_v1 import (
    _load_attr_evaluator,
    _partial_label_attr_evaluator,
)

AREA_MIN = 32.0 ** 2
AREA_MEDIUM_MAX = 96.0 ** 2
AREA_MAX = 1e5 ** 2

ANNOTATION = Path("/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json")
SOURCE_DIR = Path("/root/autodl-tmp/humanslot_outputs/qwen35_zero_shot_box_attr_v1/fullval2693_batch2_20260722")
INFERENCE = SOURCE_DIR / "inference.jsonl"
SOURCE_METRICS = SOURCE_DIR / "metrics.json"
ATTR_SOURCE = Path("/root/autodl-tmp/humanslot/HumanQueryNet_mmdet3_port/datasets/api_wrappers/eval_attr.py")
OUTPUT = SOURCE_DIR / "eval_hqnet_nosmall_20260809"


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
        indices = np.where(np.isclose(evaluator.params.iouThrs, iou))[0]
        if indices.size != 1:
            raise RuntimeError(f"cannot locate IoU threshold {iou}")
        precision = precision[indices]
    precision = precision[:, :, :, area_index, -1]
    return mean_valid(precision)


def collect_summary(evaluator: Any, aligned: bool) -> Dict[str, float]:
    if aligned:
        return {
            "AP": summarize_ap(evaluator, 0),
            "AP50": summarize_ap(evaluator, 0, 0.50),
            "AP75": summarize_ap(evaluator, 0, 0.75),
            "APm": summarize_ap(evaluator, 1),
            "APl": summarize_ap(evaluator, 2),
        }
    return {
        "AP": summarize_ap(evaluator, 0),
        "AP50": summarize_ap(evaluator, 0, 0.50),
        "AP75": summarize_ap(evaluator, 0, 0.75),
        "APs": summarize_ap(evaluator, 1),
        "APm": summarize_ap(evaluator, 2),
        "APl": summarize_ap(evaluator, 3),
    }


def run_coco(coco_gt: COCO, records: List[Dict[str, Any]], image_ids: Sequence[int], aligned: bool) -> Dict[str, float]:
    evaluator = COCOeval(coco_gt, coco_gt.loadRes(records), iouType="bbox")
    evaluator.params.imgIds = [int(value) for value in image_ids]
    if aligned:
        set_no_small_ranges(evaluator)
    evaluator.evaluate()
    evaluator.accumulate()
    return collect_summary(evaluator, aligned)


def prediction_attr(record: Dict[str, Any], name: str) -> float:
    if name != "gender":
        return float(record.get(name, -1))
    value = int(record.get("gender", -1))
    return 1.0 if value == 0 else (0.0 if value == 1 else -1.0)


def run_attr(coco_gt: COCO, predictions: List[Dict[str, Any]], name: str, image_ids: Sequence[int], aligned: bool) -> Dict[str, float]:
    records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "bbox": [float(value) for value in record["bbox"]],
            "score": float(record["score"]),
            name: prediction_attr(record, name),
        }
        for record in predictions
    ]
    evaluator_cls = _partial_label_attr_evaluator(_load_attr_evaluator(ATTR_SOURCE))
    evaluator = evaluator_cls(coco_gt, coco_gt.loadRes(records), iouType="bbox", attr_type=name)
    evaluator.params.imgIds = [int(value) for value in image_ids]
    if aligned:
        set_no_small_ranges(evaluator)
    evaluator.evaluate()
    evaluator.accumulate()
    return collect_summary(evaluator, aligned)


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
        if (bool(ann.get("iscrowd", 0)) or bool(ann.get("ignore", 0)) or len(bbox) != 4
                or float(bbox[2]) <= 0 or float(bbox[3]) <= 0
                or ("bboxes" in valid and not bool(int(valid["bboxes"])))):
            continue
        area = float(ann.get("area", float(bbox[2]) * float(bbox[3])))
        if area < AREA_MIN:
            continue
        counts["gt_persons_medium_large"] += 1
        counts["gt_persons_medium" if area < AREA_MEDIUM_MAX else "gt_persons_large"] += 1
        for name in ("gender", "age"):
            if float(ann.get(name, -1)) > -1 and (name not in valid or bool(int(valid[name]))):
                counts[f"{name}_gt_labeled_medium_large"] += 1
    return counts


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    source_metrics = json.loads(SOURCE_METRICS.read_text(encoding="utf-8"))
    rows = []
    predictions: List[Dict[str, Any]] = []
    with INFERENCE.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            rows.append(row)
            predictions.extend(row.get("records", []))

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
    ]

    standard = {
        "bbox": run_coco(coco_gt, bbox_records, image_ids, aligned=False),
        "gender": run_attr(coco_gt, predictions, "gender", image_ids, aligned=False),
        "age": run_attr(coco_gt, predictions, "age", image_ids, aligned=False),
    }
    aligned = {
        "bbox": run_coco(coco_gt, bbox_records, image_ids, aligned=True),
        "gender": run_attr(coco_gt, predictions, "gender", image_ids, aligned=True),
        "age": run_attr(coco_gt, predictions, "age", image_ids, aligned=True),
    }

    checks: Dict[str, Dict[str, Any]] = {}
    for group in ("bbox", "gender", "age"):
        for suffix in ("AP", "AP50", "AP75"):
            key = f"{group}_{suffix}"
            expected = float(source_metrics[key])
            actual = float(standard[group][suffix])
            checks[f"standard_{key}_reproduced"] = {
                "source": expected,
                "recomputed": actual,
                "absolute_difference": abs(expected - actual),
                "pass": abs(expected - actual) < 1e-12,
            }
    for suffix in ("APm", "APl"):
        key = f"bbox_{suffix}"
        expected = float(source_metrics[key])
        actual = float(aligned["bbox"][suffix])
        checks[f"aligned_{key}_matches_standard_slice"] = {
            "source": expected,
            "recomputed": actual,
            "absolute_difference": abs(expected - actual),
            "pass": abs(expected - actual) < 1e-12,
        }
    if not all(item["pass"] for item in checks.values()):
        raise RuntimeError(f"validation failed: {checks}")

    metrics: Dict[str, Any] = {
        f"{group}_{name}": value
        for group, values in aligned.items()
        for name, value in values.items()
    }
    metrics.update(valid_non_small_gt(coco_gt, image_ids))
    metrics.update({
        "images": len(image_ids),
        "inference_rows": len(rows),
        "predicted_persons": len(predictions),
        "model": source_metrics.get("model"),
        "source_prediction_dir": str(SOURCE_DIR),
        "annotation": str(ANNOTATION),
        "split": "full_val",
        "available_tasks": ["bbox", "gender", "age"],
        "unavailable_tasks": ["segm", "keypoints"],
        "score_protocol": source_metrics.get("score_protocol"),
        "area_policy": {
            "all": [AREA_MIN, AREA_MAX],
            "medium": [AREA_MIN, AREA_MEDIUM_MAX],
            "large": [AREA_MEDIUM_MAX, AREA_MAX],
            "boundary_semantics": "COCO area-range ignore rules",
        },
        "protocol": "native_qwen35_zero_shot_existing_predictions_hqnet_paper_no_small_area_aligned_cocoeval",
        "result_classification": "diagnostic; no new inference; area-aligned but confidence protocol differs from v4 generated evaluation",
    })
    manifest = {
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pycocotools": getattr(pycocotools, "__version__", "unknown"),
        "annotation": str(ANNOTATION),
        "annotation_sha256": sha256(ANNOTATION),
        "inference": str(INFERENCE),
        "inference_sha256": sha256(INFERENCE),
        "source_metrics": str(SOURCE_METRICS),
        "source_metrics_sha256": sha256(SOURCE_METRICS),
        "images": len(image_ids),
        "inference_rows": len(rows),
        "predictions_count": len(predictions),
        "seed": None,
        "deterministic": True,
        "new_inference": False,
        "metric_definition": "COCO AP with overall area range [32^2, 1e5^2], plus standard medium and large ranges",
        "aggregation": "mean valid precision over IoU, recall, and category",
        "classification": metrics["result_classification"],
        "validation_checks": checks,
    }
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUTPUT / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"metrics": metrics, "validation_checks": checks}, indent=2))


if __name__ == "__main__":
    main()
