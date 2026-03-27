from typing import Dict, List

import numpy as np


def compute_confusion_matrix(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int,
    ignore_index: int = 255,
) -> np.ndarray:
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred={pred.shape}, target={target.shape}")
    valid = target != ignore_index
    pred = pred[valid].reshape(-1)
    target = target[valid].reshape(-1)
    mask = (target >= 0) & (target < num_classes)
    pred = pred[mask]
    target = target[mask]
    idx = target * num_classes + pred
    hist = np.bincount(idx, minlength=num_classes * num_classes)
    return hist.reshape(num_classes, num_classes)


def summarize_iou(hist: np.ndarray, class_names: List[str]) -> Dict[str, object]:
    inter = np.diag(hist).astype(np.float64)
    union = hist.sum(1) + hist.sum(0) - inter
    iou = np.divide(inter, np.maximum(union, 1.0))
    acc = np.divide(inter, np.maximum(hist.sum(1), 1.0))
    mean_iou = float(iou.mean())
    mean_acc = float(acc.mean())
    pixel_acc = float(inter.sum() / max(hist.sum(), 1.0))
    per_class_iou = {name: float(iou[idx]) for idx, name in enumerate(class_names)}
    return {
        "mIoU": mean_iou,
        "mAcc": mean_acc,
        "aAcc": pixel_acc,
        "per_class_iou": per_class_iou,
    }
