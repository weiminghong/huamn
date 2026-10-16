from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


MissingSpan = Tuple[int, int, str, int]


@dataclass
class SerializedTarget:
    text: str
    box_targets: np.ndarray
    pose_targets: np.ndarray
    pose_areas: np.ndarray
    seg_targets: List[np.ndarray]
    box_person_indices: List[int]
    pose_person_indices: List[int]
    seg_person_indices: List[int]
    missing_attribute_char_spans: List[MissingSpan]
    source_ann_ids: List[int]

    @property
    def target_counts(self) -> Dict[str, int]:
        return {
            "box": len(self.box_person_indices),
            "pose": len(self.pose_person_indices),
            "seg": len(self.seg_person_indices),
        }


class _TextBuilder:
    def __init__(self) -> None:
        self.parts: List[str] = []
        self.length = 0

    def append(self, text: str) -> Tuple[int, int]:
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        return start, self.length

    def build(self) -> str:
        return "".join(self.parts)


def _normalize_box(box: Sequence[float], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return [
        min(max(x1 / width, 0.0), 1.0),
        min(max(y1 / height, 0.0), 1.0),
        min(max(x2 / width, 0.0), 1.0),
        min(max(y2 / height, 0.0), 1.0),
    ]


def _normalize_pose(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    result = np.asarray(keypoints, dtype=np.float32).copy()
    if result.shape != (17, 3):
        raise ValueError(f"expected COCO-17 keypoints, got {result.shape}")
    visible = result[:, 2] > 0
    result[visible, 0] = np.clip(result[visible, 0] / width, 0.0, 1.0)
    result[visible, 1] = np.clip(result[visible, 1] / height, 0.0, 1.0)
    result[~visible, :2] = 0
    result[:, 2] = np.clip(np.rint(result[:, 2]), 0, 2)
    return result


def serialize_target(
    instances: Sequence[Dict[str, Any]], *, width: int, height: int
) -> SerializedTarget:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    builder = _TextBuilder()
    builder.append('{"persons":[')
    boxes: List[List[float]] = []
    poses: List[np.ndarray] = []
    pose_areas: List[float] = []
    segments: List[np.ndarray] = []
    box_person_indices: List[int] = []
    pose_person_indices: List[int] = []
    seg_person_indices: List[int] = []
    missing: List[MissingSpan] = []
    source_ann_ids: List[int] = []

    for person_index, instance in enumerate(instances):
        expected_person_id = person_index + 1
        if int(instance["person_id"]) != expected_person_id:
            raise ValueError("instances must be sorted and contiguously rebound")
        if person_index:
            builder.append(",")
        builder.append('{"person_id":')
        builder.append(str(expected_person_id))
        builder.append(',"box":"<box>"')
        boxes.append(_normalize_box(instance["bbox_xyxy"], width, height))
        box_person_indices.append(person_index)

        builder.append(',"pose":')
        if instance["pose_valid"]:
            builder.append('"<pose>"')
            poses.append(_normalize_pose(instance["keypoints"], width, height))
            pose_areas.append(float(instance["area"]))
            pose_person_indices.append(person_index)
        else:
            builder.append("null")

        builder.append(',"seg":')
        if instance["seg_valid"]:
            mask = np.asarray(instance["mask"], dtype=np.bool_)
            if mask.shape != (height, width):
                raise ValueError(
                    f"mask shape {mask.shape} does not match {(height, width)}"
                )
            builder.append('"<seg>"')
            segments.append(mask)
            seg_person_indices.append(person_index)
        else:
            builder.append("null")

        builder.append(',"gender":')
        gender_text = (
            str(int(instance["gender"])) if instance["gender_valid"] else "-1"
        )
        start, end = builder.append(gender_text)
        if not instance["gender_valid"]:
            missing.append((start, end, "gender", person_index))

        builder.append(',"age":')
        age_text = (
            str(int(round(float(instance["age"]))))
            if instance["age_valid"]
            else "-1"
        )
        start, end = builder.append(age_text)
        if not instance["age_valid"]:
            missing.append((start, end, "age", person_index))
        builder.append("}")
        source_ann_ids.append(int(instance["source_ann_id"]))
    builder.append("]}")

    return SerializedTarget(
        text=builder.build(),
        box_targets=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
        pose_targets=np.asarray(poses, dtype=np.float32).reshape(-1, 17, 3),
        pose_areas=np.asarray(pose_areas, dtype=np.float32),
        seg_targets=segments,
        box_person_indices=box_person_indices,
        pose_person_indices=pose_person_indices,
        seg_person_indices=seg_person_indices,
        missing_attribute_char_spans=missing,
        source_ann_ids=source_ann_ids,
    )

