from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from .schema_v4 import BOX_SCALE, POSE_TOKEN, SEG_TOKEN


TEXT_TASKS = (
    "structure",
    "box",
    "pose_route",
    "seg_route",
    "gender",
    "age",
)
TEXT_TASK_TO_ID = {name: index + 1 for index, name in enumerate(TEXT_TASKS)}
IGNORE_TASK_ID = 0


@dataclass(frozen=True)
class TargetSpan:
    start: int
    end: int
    task: str
    person_index: int
    valid: bool = True


@dataclass
class SerializedTargetV4:
    text: str
    spans: List[TargetSpan]
    pose_targets: np.ndarray
    pose_areas: np.ndarray
    seg_targets: List[np.ndarray]
    pose_person_indices: List[int]
    seg_person_indices: List[int]
    source_ann_ids: List[int]

    @property
    def person_count(self) -> int:
        return len(self.source_ann_ids)


class _Builder:
    def __init__(self) -> None:
        self.parts: List[str] = []
        self.spans: List[TargetSpan] = []
        self.length = 0

    def append(
        self,
        text: str,
        task: str = "structure",
        person_index: int = -1,
        valid: bool = True,
    ) -> None:
        if task not in TEXT_TASK_TO_ID:
            raise ValueError(f"unknown text task: {task}")
        start = self.length
        self.parts.append(text)
        self.length += len(text)
        if text:
            self.spans.append(
                TargetSpan(start, self.length, task, person_index, valid)
            )

    def build(self) -> str:
        return "".join(self.parts)


def quantize_box(
    box: Sequence[float], width: int, height: int, scale: int = BOX_SCALE
) -> List[int]:
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in box]
    normalized = np.asarray(
        [x1 / width, y1 / height, x2 / width, y2 / height],
        dtype=np.float64,
    )
    quantized = np.rint(np.clip(normalized, 0.0, 1.0) * scale).astype(np.int64)
    quantized[2] = max(int(quantized[2]), min(scale, int(quantized[0]) + 1))
    quantized[3] = max(int(quantized[3]), min(scale, int(quantized[1]) + 1))
    return [int(value) for value in quantized]


def dequantize_box(
    box: Sequence[int], width: int, height: int, scale: int = BOX_SCALE
) -> List[float]:
    x1, y1, x2, y2 = [int(value) for value in box]
    return [
        x1 * width / scale,
        y1 * height / scale,
        x2 * width / scale,
        y2 * height / scale,
    ]


def normalize_pose(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    result = np.asarray(keypoints, dtype=np.float32).copy()
    if result.shape != (17, 3):
        raise ValueError(f"expected COCO-17 keypoints, got {result.shape}")
    visible = result[:, 2] > 0
    result[visible, 0] = np.clip(result[visible, 0] / width, 0.0, 1.0)
    result[visible, 1] = np.clip(result[visible, 1] / height, 0.0, 1.0)
    result[~visible, :2] = 0.0
    result[:, 2] = np.clip(np.rint(result[:, 2]), 0, 2)
    return result


def serialize_target_v4(
    instances: Sequence[Dict[str, Any]], *, width: int, height: int
) -> SerializedTargetV4:
    builder = _Builder()
    builder.append('{"persons":[')
    poses: List[np.ndarray] = []
    pose_areas: List[float] = []
    segments: List[np.ndarray] = []
    pose_person_indices: List[int] = []
    seg_person_indices: List[int] = []
    source_ann_ids: List[int] = []

    for person_index, instance in enumerate(instances):
        expected_id = person_index + 1
        if int(instance["person_id"]) != expected_id:
            raise ValueError("instances must be deterministically sorted and rebound")
        if person_index:
            builder.append(",")
        builder.append('{"person_id":')
        builder.append(str(expected_id))
        builder.append(',"box":')
        box = quantize_box(instance["bbox_xyxy"], width, height)
        builder.append(
            "[" + ",".join(str(value) for value in box) + "]",
            "box",
            person_index,
        )
        builder.append(',"pose":"')
        builder.append(POSE_TOKEN, "pose_route", person_index)
        builder.append('","seg":"')
        builder.append(SEG_TOKEN, "seg_route", person_index)
        builder.append('","gender":')
        gender_valid = bool(instance["gender_valid"])
        gender = int(instance["gender"]) if gender_valid else -1
        builder.append(str(gender), "gender", person_index, gender_valid)
        builder.append(',"age":')
        age_valid = bool(instance["age_valid"])
        age = int(round(float(instance["age"]))) if age_valid else -1
        age = min(max(age, -1), 120)
        builder.append(str(age), "age", person_index, age_valid)
        builder.append("}")

        if bool(instance["pose_valid"]):
            poses.append(normalize_pose(instance["keypoints"], width, height))
            pose_areas.append(float(instance["area"]))
            pose_person_indices.append(person_index)
        if bool(instance["seg_valid"]):
            mask = np.asarray(instance["mask"], dtype=np.bool_)
            if mask.shape != (height, width):
                raise ValueError(
                    f"mask shape {mask.shape} does not match {(height, width)}"
                )
            segments.append(mask)
            seg_person_indices.append(person_index)
        source_ann_ids.append(int(instance["source_ann_id"]))

    builder.append("]}")
    return SerializedTargetV4(
        text=builder.build(),
        spans=builder.spans,
        pose_targets=np.asarray(poses, dtype=np.float32).reshape(-1, 17, 3),
        pose_areas=np.asarray(pose_areas, dtype=np.float32),
        seg_targets=segments,
        pose_person_indices=pose_person_indices,
        seg_person_indices=seg_person_indices,
        source_ann_ids=source_ann_ids,
    )


def assign_token_tasks(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[TargetSpan]
) -> Tuple[List[int], List[bool]]:
    task_ids: List[int] = []
    valid_flags: List[bool] = []
    for token_start, token_end in offsets:
        if token_end <= token_start:
            task_ids.append(IGNORE_TASK_ID)
            valid_flags.append(False)
            continue
        candidates = []
        for span in spans:
            overlap = max(
                0, min(token_end, span.end) - max(token_start, span.start)
            )
            if overlap:
                preference = int(span.task != "structure")
                candidates.append((overlap, preference, span))
        if not candidates:
            task_ids.append(IGNORE_TASK_ID)
            valid_flags.append(False)
            continue
        _, _, selected = max(candidates, key=lambda value: (value[0], value[1]))
        task_ids.append(TEXT_TASK_TO_ID[selected.task])
        valid_flags.append(bool(selected.valid))
    return task_ids, valid_flags
