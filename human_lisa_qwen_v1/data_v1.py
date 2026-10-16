from __future__ import annotations

import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset


COCO_FLIP_INDEX = np.asarray(
    [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15],
    dtype=np.int64,
)


def _as_bool(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def _finite_number(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _valid_box(annotation: Dict[str, Any]) -> bool:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    values = [_finite_number(value) for value in bbox]
    if any(value is None for value in values):
        return False
    _, _, width, height = values
    area = _finite_number(annotation.get("area", width * height))
    return bool(width > 0 and height > 0 and area is not None and area > 0)


def decode_segmentation(
    segmentation: Any, height: int, width: int
) -> Optional[np.ndarray]:
    if not segmentation:
        return None
    try:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, int(height), int(width))
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict):
            rle = segmentation
            if isinstance(rle.get("counts"), list):
                rle = mask_utils.frPyObjects(rle, int(height), int(width))
        else:
            return None
        decoded = mask_utils.decode(rle)
    except (TypeError, ValueError, IndexError):
        return None
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    mask = np.asarray(decoded, dtype=np.bool_)
    if mask.shape != (int(height), int(width)):
        return None
    return mask


def _pose_from_annotation(annotation: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    valid = annotation.get("valid") or {}
    values = annotation.get("keypoints")
    pose_valid = _as_bool(valid.get("keypoints_2d", 0))
    try:
        num_keypoints = int(annotation.get("num_keypoints", 0))
    except (TypeError, ValueError):
        num_keypoints = 0
    if (
        not pose_valid
        or num_keypoints <= 0
        or not isinstance(values, list)
        or len(values) < 51
    ):
        return np.zeros((17, 3), dtype=np.float32), False
    keypoints = np.asarray(values[:51], dtype=np.float32).reshape(17, 3)
    if not np.isfinite(keypoints).all() or not np.any(keypoints[:, 2] > 0):
        return np.zeros((17, 3), dtype=np.float32), False
    return keypoints, True


def _attribute(
    annotation: Dict[str, Any], name: str, default: float
) -> Tuple[float, bool]:
    valid = annotation.get("valid") or {}
    is_valid = _as_bool(valid.get(name, 0))
    value = _finite_number(annotation.get(name)) if is_valid else None
    if value is None:
        return float(default), False
    return float(value), True


def _sort_and_rebind(instances: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = list(instances)
    result.sort(
        key=lambda item: (
            (item["bbox_xyxy"][0] + item["bbox_xyxy"][2]) * 0.5,
            (item["bbox_xyxy"][1] + item["bbox_xyxy"][3]) * 0.5,
            int(item["source_ann_id"]),
        )
    )
    for person_id, item in enumerate(result, start=1):
        item["person_id"] = person_id
    return result


def build_instances(
    annotations: Sequence[Dict[str, Any]],
    *,
    width: int,
    height: int,
    include_eval_segmentation: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    instances: List[Dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for annotation in annotations:
        if _as_bool(annotation.get("ignore", 0)):
            skipped["ignored"] += 1
            continue
        if _as_bool(annotation.get("iscrowd", 0)):
            skipped["crowd"] += 1
            continue
        valid = annotation.get("valid") or {}
        if not _as_bool(valid.get("bboxes", 1)) or not _valid_box(annotation):
            skipped["invalid_box"] += 1
            continue

        x, y, box_width, box_height = [
            float(value) for value in annotation["bbox"]
        ]
        x1 = max(0.0, min(x, float(width)))
        y1 = max(0.0, min(y, float(height)))
        x2 = max(0.0, min(x + box_width, float(width)))
        y2 = max(0.0, min(y + box_height, float(height)))
        if x2 <= x1 or y2 <= y1:
            skipped["invalid_box"] += 1
            continue

        keypoints, pose_valid = _pose_from_annotation(annotation)
        seg_present = bool(annotation.get("segmentation"))
        seg_valid = seg_present and (
            _as_bool(valid.get("segmentation", 0))
            or bool(include_eval_segmentation)
        )
        mask = (
            decode_segmentation(annotation.get("segmentation"), height, width)
            if seg_valid
            else None
        )
        if seg_valid and mask is None:
            seg_valid = False
            skipped["invalid_segmentation"] += 1

        gender, gender_valid = _attribute(annotation, "gender", -1.0)
        age, age_valid = _attribute(annotation, "age", -1.0)
        instance = {
            "source_ann_id": int(annotation["id"]),
            "source_image_id": int(annotation["image_id"]),
            "bbox_xyxy": [x1, y1, x2, y2],
            "area": float(annotation.get("area", (x2 - x1) * (y2 - y1))),
            "keypoints": keypoints,
            "pose_valid": bool(pose_valid),
            "mask": mask,
            "seg_valid": bool(seg_valid),
            "gender": int(round(gender)) if gender_valid else -1,
            "gender_valid": bool(gender_valid),
            "age": float(age) if age_valid else -1.0,
            "age_valid": bool(age_valid),
        }
        instances.append(instance)
    return _sort_and_rebind(instances), dict(skipped)


def horizontal_flip_instances(
    instances: Sequence[Dict[str, Any]], *, width: int
) -> List[Dict[str, Any]]:
    flipped: List[Dict[str, Any]] = []
    for source in instances:
        item = copy.deepcopy(source)
        x1, y1, x2, y2 = item["bbox_xyxy"]
        item["bbox_xyxy"] = [
            float(width) - float(x2),
            float(y1),
            float(width) - float(x1),
            float(y2),
        ]
        if item["pose_valid"]:
            keypoints = np.asarray(item["keypoints"], dtype=np.float32).copy()
            keypoints[:, 0] = float(width) - keypoints[:, 0]
            item["keypoints"] = keypoints[COCO_FLIP_INDEX].copy()
        if item["seg_valid"] and item["mask"] is not None:
            item["mask"] = np.fliplr(item["mask"]).copy()
        flipped.append(item)
    return _sort_and_rebind(flipped)


class CocoUniHumanV1Dataset(Dataset):
    def __init__(
        self,
        annotation_file: str,
        image_root: str,
        *,
        training: bool,
        horizontal_flip_prob: float = 0.0,
        max_samples: Optional[int] = None,
        decode_eval_segmentation: bool = True,
    ) -> None:
        with open(annotation_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.annotation_file = str(annotation_file)
        self.image_root = Path(image_root)
        self.training = bool(training)
        self.horizontal_flip_prob = float(horizontal_flip_prob)
        self.decode_eval_segmentation = bool(decode_eval_segmentation)
        if not 0.0 <= self.horizontal_flip_prob <= 1.0:
            raise ValueError("horizontal_flip_prob must be in [0, 1]")
        self.images = list(payload.get("images", []))
        if max_samples is not None:
            self.images = self.images[: int(max_samples)]
        selected_ids = {int(image["id"]) for image in self.images}
        annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation["image_id"])
            if image_id in selected_ids:
                annotations_by_image[image_id].append(annotation)
        self.annotations_by_image = dict(annotations_by_image)
        self.categories = list(payload.get("categories", []))

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        image_path = self.image_root / str(image_info["file_name"])
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        instances, skipped = build_instances(
            self.annotations_by_image.get(image_id, []),
            width=width,
            height=height,
            include_eval_segmentation=(
                not self.training and self.decode_eval_segmentation
            ),
        )
        flipped = (
            self.training
            and self.horizontal_flip_prob > 0
            and random.random() < self.horizontal_flip_prob
        )
        if flipped:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            instances = horizontal_flip_instances(instances, width=width)
        return {
            "id": image_id,
            "file_name": str(image_info["file_name"]),
            "image_path": str(image_path),
            "image": image,
            "width": width,
            "height": height,
            "instances": instances,
            "skipped": skipped,
            "flipped": bool(flipped),
        }


def audit_annotation_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    images = payload.get("images", [])
    annotations = payload.get("annotations", [])
    valid_counts: Counter[str] = Counter()
    noncrowd_counts: Counter[int] = Counter()
    invalid_box = 0
    present_segmentation = 0
    for annotation in annotations:
        for name, value in (annotation.get("valid") or {}).items():
            valid_counts[name] += int(_as_bool(value))
        present_segmentation += int(bool(annotation.get("segmentation")))
        if not _valid_box(annotation):
            invalid_box += 1
        if (
            not _as_bool(annotation.get("iscrowd", 0))
            and not _as_bool(annotation.get("ignore", 0))
            and _valid_box(annotation)
        ):
            noncrowd_counts[int(annotation["image_id"])] += 1
    return {
        "images": len(images),
        "annotations": len(annotations),
        "invalid_box": invalid_box,
        "present_segmentation": present_segmentation,
        "valid": dict(valid_counts),
        "max_noncrowd_per_image": max(noncrowd_counts.values(), default=0),
    }
