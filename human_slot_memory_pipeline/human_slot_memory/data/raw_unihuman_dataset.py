from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from torch.utils.data import Dataset

from ..schema import HumanBatch, HumanSample, PersonSlot, TaskMask


def _normalize_keypoints(flat_keypoints: List[float]) -> Optional[List[List[float]]]:
    if not isinstance(flat_keypoints, list) or len(flat_keypoints) != 51:
        return None
    keypoints = []
    visible = False
    for idx in range(0, len(flat_keypoints), 3):
        x, y, v = flat_keypoints[idx : idx + 3]
        point = [float(x), float(y), float(v)]
        keypoints.append(point)
        visible = visible or point[2] > 0
    return keypoints if visible else None


def _normalize_box(coco_bbox: List[float]) -> Optional[List[float]]:
    if not isinstance(coco_bbox, list) or len(coco_bbox) != 4:
        return None
    x, y, w, h = [float(value) for value in coco_bbox]
    return [x, y, x + w, y + h]


def _build_slot(annotation: Dict[str, Any], slot_id: int) -> PersonSlot:
    valid = annotation.get("valid", {}) if isinstance(annotation.get("valid"), dict) else {}
    box_xyxy = _normalize_box(annotation.get("bbox"))
    pose = _normalize_keypoints(annotation.get("keypoints", []))
    attributes: Dict[str, Any] = {}
    if "gender" in annotation:
        attributes["gender"] = annotation["gender"]
    if "age" in annotation:
        attributes["age"] = annotation["age"]

    task_mask = TaskMask(
        box=bool(valid.get("bboxes", 1)) and box_xyxy is not None,
        pose=bool(valid.get("keypoints_2d", 0)) and pose is not None,
        attr=bool(valid.get("gender", 0) or valid.get("age", 0)) and bool(attributes),
    )
    return PersonSlot(
        slot_id=slot_id,
        box_xyxy=box_xyxy,
        pose17_xyv=pose,
        attributes=attributes,
        task_mask=task_mask,
    )


class RawCocoUniHumanDataset(Dataset):
    def __init__(self, annotation_path: str | Path, image_root: str | Path, max_samples: Optional[int] = None) -> None:
        with open(annotation_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        self.image_root = Path(image_root)
        self.images = payload.get("images", [])
        self.annotations_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            image_id = int(annotation.get("image_id", -1))
            if image_id >= 0:
                self.annotations_by_image[image_id].append(annotation)
        if max_samples is not None:
            self.images = self.images[: int(max_samples)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> HumanSample:
        image_info = self.images[index]
        image_id = int(image_info["id"])
        annotations = self.annotations_by_image.get(image_id, [])
        slots = [_build_slot(annotation, slot_id=slot_idx) for slot_idx, annotation in enumerate(annotations)]
        image_name = image_info.get("file_name") or image_info.get("image") or ""
        return HumanSample(
            sample_id=str(image_id),
            image_path=str(self.image_root / image_name),
            width=int(image_info.get("width", 0) or 0),
            height=int(image_info.get("height", 0) or 0),
            slots=slots,
            raw={"image": image_info, "annotations": annotations},
        )


def human_collate(samples: Iterable[HumanSample]) -> HumanBatch:
    return HumanBatch(samples=list(samples))
