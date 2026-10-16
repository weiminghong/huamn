from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class Match:
    prediction_index: int
    target_index: int
    iou: float


def box_iou_matrix(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
) -> np.ndarray:
    pred = np.asarray(predictions, dtype=np.float64).reshape(-1, 4)
    target = np.asarray(targets, dtype=np.float64).reshape(-1, 4)
    if not len(pred) or not len(target):
        return np.zeros((len(pred), len(target)), dtype=np.float64)
    top_left = np.maximum(pred[:, None, :2], target[None, :, :2])
    bottom_right = np.minimum(pred[:, None, 2:], target[None, :, 2:])
    intersection_size = np.clip(bottom_right - top_left, 0.0, None)
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    pred_area = np.prod(np.clip(pred[:, 2:] - pred[:, :2], 0.0, None), axis=1)
    target_area = np.prod(
        np.clip(target[:, 2:] - target[:, :2], 0.0, None), axis=1
    )
    union = pred_area[:, None] + target_area[None] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def hungarian_box_match(
    predictions: Sequence[Sequence[float]],
    targets: Sequence[Sequence[float]],
    *,
    minimum_iou: float = 0.5,
) -> List[Match]:
    if not 0.0 <= minimum_iou <= 1.0:
        raise ValueError("minimum_iou must be in [0, 1]")
    ious = box_iou_matrix(predictions, targets)
    if not ious.size:
        return []
    prediction_indices, target_indices = linear_sum_assignment(1.0 - ious)
    matches = []
    for prediction_index, target_index in zip(
        prediction_indices.tolist(), target_indices.tolist()
    ):
        iou = float(ious[prediction_index, target_index])
        if iou >= minimum_iou:
            matches.append(Match(prediction_index, target_index, iou))
    return matches
