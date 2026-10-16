from __future__ import annotations

import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def encode_binary_mask(mask: np.ndarray) -> Dict[str, Any]:
    array = np.asfortranarray(np.asarray(mask, dtype=np.uint8))
    if array.ndim != 2:
        raise ValueError("binary mask must be HxW")
    encoded = mask_utils.encode(array)
    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")
    return {
        "size": [int(value) for value in encoded["size"]],
        "counts": str(counts),
    }


def _load_results(coco_gt: COCO, records: List[Dict[str, Any]]) -> COCO:
    coco_gt.dataset.setdefault("info", {})
    return coco_gt.loadRes(records)


def _run_coco_eval(
    coco_gt: COCO,
    records: List[Dict[str, Any]],
    *,
    iou_type: str,
    image_ids: Sequence[int],
) -> Dict[str, float]:
    if not records:
        if iou_type == "keypoints":
            return {
                "AP": 0.0,
                "AP50": 0.0,
                "AP75": 0.0,
                "APm": 0.0,
                "APl": 0.0,
            }
        return {
            "AP": 0.0,
            "AP50": 0.0,
            "AP75": 0.0,
            "APs": 0.0,
            "APm": 0.0,
            "APl": 0.0,
        }
    coco_dt = _load_results(coco_gt, records)
    evaluator = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    evaluator.params.imgIds = [int(value) for value in image_ids]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    if iou_type == "keypoints":
        return {
            "AP": float(stats[0]),
            "AP50": float(stats[1]),
            "AP75": float(stats[2]),
            "APm": float(stats[3]),
            "APl": float(stats[4]),
        }
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
        "APs": float(stats[3]),
        "APm": float(stats[4]),
        "APl": float(stats[5]),
    }


def _load_attr_evaluator(source: Path):
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"AttrCOCOeval source not found: {source}")
    spec = importlib.util.spec_from_file_location(
        "humanquerynet_eval_attr_v1", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load AttrCOCOeval from {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AttrCOCOeval


def _partial_label_attr_evaluator(base_class):
    class PartialLabelAttrCOCOeval(base_class):
        """Ignore detections on people without this attribute annotation."""
        def _prepare(self):
            params = self.params
            if params.useCats:
                gts = self.cocoGt.loadAnns(
                    self.cocoGt.getAnnIds(imgIds=params.imgIds, catIds=params.catIds)
                )
                dts = self.cocoDt.loadAnns(
                    self.cocoDt.getAnnIds(imgIds=params.imgIds, catIds=params.catIds)
                )
            else:
                gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=params.imgIds))
                dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=params.imgIds))
            valid_images = set()
            for gt in gts:
                valid = gt.get("valid", {})
                attr_valid = (
                    float(gt.get(self.attr_type, -1)) > -1
                    and (self.attr_type not in valid or bool(int(valid[self.attr_type])))
                )
                gt["ignore"] = int(bool(gt.get("iscrowd", 0)) or not attr_valid)
                if attr_valid:
                    valid_images.add(int(gt["image_id"]))
            self.img_ids = np.asarray(sorted(valid_images), dtype=np.int64)
            self.params.imgIds = self.img_ids
            gts = [gt for gt in gts if int(gt["image_id"]) in valid_images]
            dts = [dt for dt in dts if int(dt["image_id"]) in valid_images]
            self._gts = defaultdict(list)
            self._dts = defaultdict(list)
            for gt in gts:
                self._gts[gt["image_id"], gt["category_id"]].append(gt)
            for dt in dts:
                self._dts[dt["image_id"], dt["category_id"]].append(dt)
            self.evalImgs = defaultdict(list)
            self.eval = {}

    return PartialLabelAttrCOCOeval


def _run_attribute_eval(
    coco_gt: COCO,
    records: List[Dict[str, Any]],
    *,
    attr_name: str,
    image_ids: Sequence[int],
    source: Path,
) -> Dict[str, float]:
    def prediction_value(record: Dict[str, Any]) -> float:
        if attr_name != "gender":
            return float(record.get(attr_name, -1))
        if "gender_score" in record:
            return float(record["gender_score"])
        gender_class = int(record.get("gender", -1))
        if gender_class == 0:
            return 1.0
        if gender_class == 1:
            return 0.0
        return -1.0

    attr_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "bbox": [float(value) for value in record["bbox"]],
            "score": float(record["score"]),
            attr_name: prediction_value(record),
        }
        for record in records
        if "bbox" in record
    ]
    if not attr_records:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}
    AttrCOCOeval = _partial_label_attr_evaluator(_load_attr_evaluator(source))
    coco_dt = _load_results(coco_gt, attr_records)
    evaluator = AttrCOCOeval(
        coco_gt, coco_dt, iouType="bbox", attr_type=attr_name
    )
    evaluator.params.imgIds = [int(value) for value in image_ids]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    stats = evaluator.stats
    return {
        "AP": float(stats[0]),
        "AP50": float(stats[1]),
        "AP75": float(stats[2]),
    }


def _coverage_diagnostics(
    coco_gt: COCO, image_ids: Sequence[int]
) -> Dict[str, float]:
    selected = {int(value) for value in image_ids}
    persons = []
    for annotation in coco_gt.dataset.get("annotations", []):
        if int(annotation.get("image_id", -1)) not in selected:
            continue
        if bool(annotation.get("iscrowd", 0)) or bool(annotation.get("ignore", 0)):
            continue
        bbox = annotation.get("bbox", [])
        valid = annotation.get("valid") or {}
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or float(bbox[2]) <= 0
            or float(bbox[3]) <= 0
            or float(annotation.get("area", bbox[2] * bbox[3])) <= 0
            or ("bboxes" in valid and not bool(int(valid["bboxes"])))
        ):
            continue
        persons.append(annotation)
    total = len(persons)
    result: Dict[str, float] = {"gt_persons": int(total)}
    for name in ("gender", "age"):
        labeled = [
            annotation
            for annotation in persons
            if float(annotation.get(name, -1)) > -1
        ]
        result[f"{name}_gt_labeled"] = int(len(labeled))
        result[f"{name}_gt_coverage"] = len(labeled) / max(1, total)
        result[f"{name}_eval_images"] = len(
            {int(annotation["image_id"]) for annotation in labeled}
        )
    return result


def evaluate_all_official(
    annotation_file: str,
    predictions: Sequence[Dict[str, Any]],
    *,
    image_ids: Optional[Sequence[int]] = None,
    attr_eval_source: Path,
) -> Dict[str, float]:
    coco_gt = COCO(annotation_file)
    selected_ids = (
        sorted(int(value) for value in image_ids)
        if image_ids is not None
        else sorted(coco_gt.getImgIds())
    )
    prediction_list = [dict(record) for record in predictions]
    coverage_metrics = _coverage_diagnostics(coco_gt, selected_ids)
    bbox_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "bbox": [float(value) for value in record["bbox"]],
            "score": float(record["score"]),
        }
        for record in prediction_list
        if "bbox" in record
    ]
    keypoint_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "keypoints": [float(value) for value in record["keypoints"]],
            "score": float(record["score"]),
        }
        for record in prediction_list
        if "keypoints" in record and len(record["keypoints"]) == 51
    ]
    segm_records = [
        {
            "image_id": int(record["image_id"]),
            "category_id": int(record.get("category_id", 1)),
            "segmentation": record["segmentation"],
            "score": float(record["score"]),
        }
        for record in prediction_list
        if "segmentation" in record
    ]

    metric_groups = {
        "bbox": _run_coco_eval(
            coco_gt, bbox_records, iou_type="bbox", image_ids=selected_ids
        ),
        "keypoints": _run_coco_eval(
            coco_gt,
            keypoint_records,
            iou_type="keypoints",
            image_ids=selected_ids,
        ),
        "segm": _run_coco_eval(
            coco_gt, segm_records, iou_type="segm", image_ids=selected_ids
        ),
        "gender": _run_attribute_eval(
            coco_gt,
            prediction_list,
            attr_name="gender",
            image_ids=selected_ids,
            source=attr_eval_source,
        ),
        "age": _run_attribute_eval(
            coco_gt,
            prediction_list,
            attr_name="age",
            image_ids=selected_ids,
            source=attr_eval_source,
        ),
    }
    metrics = {
        f"{group}_{name}": value
        for group, group_metrics in metric_groups.items()
        for name, value in group_metrics.items()
    }
    metrics.update(coverage_metrics)
    return metrics

