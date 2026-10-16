from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw


COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4),
]


class MonitoredStage1Collator:
    """Attach only cheap image metadata; pixels are reopened on monitor steps."""

    def __init__(self, base_collator: Any) -> None:
        self.base_collator = base_collator

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = self.base_collator(samples)
        batch["monitor_image_paths"] = [str(sample["image_path"]) for sample in samples]
        batch["monitor_file_names"] = [str(sample["file_name"]) for sample in samples]
        batch["monitor_flipped"] = [bool(sample.get("flipped", False)) for sample in samples]
        return batch


def snapshot_predictions(predictions: Sequence[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    snapshots: List[Dict[str, torch.Tensor]] = []
    for prediction in predictions:
        item: Dict[str, torch.Tensor] = {}
        for name, value in prediction.items():
            detached = value.detach().cpu()
            item[name] = detached.float() if detached.is_floating_point() else detached
        snapshots.append(item)
    return snapshots


def _box_iou_aligned(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if predicted.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)
    top_left = torch.maximum(predicted[:, :2], target[:, :2])
    bottom_right = torch.minimum(predicted[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    predicted_area = (predicted[:, 2:] - predicted[:, :2]).clamp(min=0).prod(dim=-1)
    target_area = (target[:, 2:] - target[:, :2]).clamp(min=0).prod(dim=-1)
    return intersection / (predicted_area + target_area - intersection).clamp(min=1e-7)


def _resize_masks(logits: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if logits.shape[0] == 0:
        return torch.zeros((0, height, width), dtype=torch.bool)
    resized = F.interpolate(
        logits.float().unsqueeze(1),
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    return resized.sigmoid() >= 0.5


def _sample_metrics(
    raw_batch: Dict[str, Any],
    prediction: Dict[str, torch.Tensor],
    batch_index: int,
) -> Dict[str, Any]:
    height, width = raw_batch["original_sizes"][batch_index]
    target_boxes = raw_batch["box_targets"][batch_index].float()
    predicted_boxes = prediction["matched_query_boxes_xyxy"].float()
    box_iou = _box_iou_aligned(predicted_boxes, target_boxes)

    pose_target = raw_batch["pose_targets"][batch_index].float()
    pose_prediction = prediction["pose_xy"].float()
    pose_visibility_logits = prediction["pose_visibility"].float()
    pose_indices = raw_batch["pose_person_indices"][batch_index].long()
    pose_visible = pose_target[..., 2] > 0 if pose_target.numel() else torch.zeros((0, 17), dtype=torch.bool)
    pose_visible_count = int(pose_visible.sum().item())
    pose_pck = None
    pose_error = None
    visibility_accuracy = None
    predicted_spread = None
    target_spread = None
    if pose_target.shape[0]:
        pose_boxes = target_boxes.index_select(0, pose_indices)
        pose_scale = torch.maximum(
            pose_boxes[:, 2] - pose_boxes[:, 0],
            pose_boxes[:, 3] - pose_boxes[:, 1],
        ).clamp(min=1e-4)
        distance = (pose_prediction - pose_target[..., :2]).square().sum(dim=-1).sqrt()
        distance = distance / pose_scale[:, None]
        if pose_visible_count:
            pose_pck = float((distance[pose_visible] <= 0.2).float().mean().item())
            pose_error = float(distance[pose_visible].mean().item())
        target_visibility = pose_target[..., 2].long().clamp(0, 2)
        predicted_visibility = pose_visibility_logits.argmax(dim=-1)
        visibility_accuracy = float((predicted_visibility == target_visibility).float().mean().item())
        if pose_target.shape[0] > 1:
            predicted_spread = float(pose_prediction.flatten(1).std(dim=0, unbiased=False).mean().item())
            target_spread = float(pose_target[..., :2].flatten(1).std(dim=0, unbiased=False).mean().item())

    seg_target = raw_batch["seg_targets"][batch_index].bool()
    seg_prediction = _resize_masks(prediction["mask_logits"], int(height), int(width))
    seg_ious: List[float] = []
    seg_dices: List[float] = []
    if seg_target.shape[0]:
        intersection = (seg_prediction & seg_target).flatten(1).sum(dim=1).float()
        union = (seg_prediction | seg_target).flatten(1).sum(dim=1).float().clamp(min=1)
        denominator = seg_prediction.flatten(1).sum(dim=1) + seg_target.flatten(1).sum(dim=1)
        seg_ious = [float(value) for value in (intersection / union).tolist()]
        seg_dices = [float(value) for value in ((2.0 * intersection) / denominator.clamp(min=1)).tolist()]

    return {
        "image_id": int(raw_batch["sample_ids"][batch_index]),
        "file_name": raw_batch["monitor_file_names"][batch_index],
        "flipped": bool(raw_batch["monitor_flipped"][batch_index]),
        "persons": int(raw_batch["num_persons"][batch_index]),
        "pose_people": int(pose_target.shape[0]),
        "pose_visible_keypoints": pose_visible_count,
        "pose_pck@0.2_gt_box": pose_pck,
        "pose_mean_normalized_error_gt_box": pose_error,
        "pose_visibility_accuracy": visibility_accuracy,
        "pose_predicted_person_spread": predicted_spread,
        "pose_target_person_spread": target_spread,
        "seg_people": int(seg_target.shape[0]),
        "seg_binary_miou": sum(seg_ious) / len(seg_ious) if seg_ious else None,
        "seg_binary_dice": sum(seg_dices) / len(seg_dices) if seg_dices else None,
        "query_box_mean_iou": float(box_iou.mean().item()) if box_iou.numel() else None,
        "query_box_iou50_rate": float((box_iou >= 0.5).float().mean().item()) if box_iou.numel() else None,
    }


def _normalized_point(point: torch.Tensor, width: int, height: int) -> tuple[int, int]:
    x = max(0.0, min(1.0, float(point[0]))) * max(1, width - 1)
    y = max(0.0, min(1.0, float(point[1]))) * max(1, height - 1)
    return int(round(x)), int(round(y))


def _draw_box(draw: ImageDraw.ImageDraw, box: torch.Tensor, width: int, height: int, color: tuple[int, int, int], line_width: int) -> None:
    x1, y1 = _normalized_point(box[:2], width, height)
    x2, y2 = _normalized_point(box[2:], width, height)
    draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)


def _draw_pose(
    draw: ImageDraw.ImageDraw,
    pose: torch.Tensor,
    visibility: torch.Tensor,
    width: int,
    height: int,
    color: tuple[int, int, int],
    line_width: int,
    radius: int,
    require_visible_edges: bool,
) -> None:
    points = [_normalized_point(point, width, height) for point in pose]
    for first, second in COCO_SKELETON:
        if require_visible_edges and (int(visibility[first]) <= 0 or int(visibility[second]) <= 0):
            continue
        draw.line((points[first], points[second]), fill=color, width=line_width)
    for point_index, (x, y) in enumerate(points):
        if require_visible_edges and int(visibility[point_index]) <= 0:
            continue
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=color, width=max(1, line_width))


def _render_sample(
    raw_batch: Dict[str, Any],
    prediction: Dict[str, torch.Tensor],
    batch_index: int,
    metrics: Dict[str, Any],
    output_path: Path,
) -> None:
    with Image.open(raw_batch["monitor_image_paths"][batch_index]) as source:
        image = source.convert("RGB")
    if raw_batch["monitor_flipped"][batch_index]:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    width, height = image.size

    seg_target = raw_batch["seg_targets"][batch_index].bool()
    seg_prediction = _resize_masks(prediction["mask_logits"], height, width)
    if seg_target.shape[0]:
        target_union = seg_target.any(dim=0).numpy()
        predicted_union = seg_prediction.any(dim=0).numpy()
        pixels = np.asarray(image, dtype=np.float32).copy()
        target_only = target_union & ~predicted_union
        predicted_only = predicted_union & ~target_union
        overlap = target_union & predicted_union
        alpha = 0.38
        for mask, color in [
            (target_only, np.array([30, 210, 70], dtype=np.float32)),
            (predicted_only, np.array([225, 45, 190], dtype=np.float32)),
            (overlap, np.array([245, 210, 35], dtype=np.float32)),
        ]:
            pixels[mask] = pixels[mask] * (1.0 - alpha) + color * alpha
        image = Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")

    draw = ImageDraw.Draw(image)
    target_boxes = raw_batch["box_targets"][batch_index].float()
    predicted_boxes = prediction["matched_query_boxes_xyxy"].float()
    for box in target_boxes:
        _draw_box(draw, box, width, height, (20, 210, 60), 3)
    for box in predicted_boxes:
        _draw_box(draw, box, width, height, (230, 45, 45), 2)

    pose_target = raw_batch["pose_targets"][batch_index].float()
    pose_prediction = prediction["pose_xy"].float()
    pose_visibility = prediction["pose_visibility"].argmax(dim=-1)
    for person_index in range(pose_target.shape[0]):
        _draw_pose(
            draw, pose_target[person_index, :, :2], pose_target[person_index, :, 2],
            width, height, (20, 220, 80), 3, 3, True,
        )
        _draw_pose(
            draw, pose_prediction[person_index], pose_visibility[person_index],
            width, height, (240, 45, 45), 2, 2, False,
        )

    pck = metrics["pose_pck@0.2_gt_box"]
    miou = metrics["seg_binary_miou"]
    caption = (
        f"id={metrics['image_id']} persons={metrics['persons']} "
        f"posePCK={pck:.3f}" if pck is not None else f"id={metrics['image_id']} persons={metrics['persons']} posePCK=n/a"
    )
    caption += f" segIoU={miou:.3f}" if miou is not None else " segIoU=n/a"
    caption += " | GT=green Pred=red MaskOverlap=yellow"
    text_box = draw.textbbox((0, 0), caption)
    draw.rectangle((4, 4, min(width - 1, text_box[2] + 12), text_box[3] + 12), fill=(0, 0, 0))
    draw.text((8, 7), caption, fill=(255, 255, 255))
    image.save(output_path)


def write_training_monitor(
    *,
    output_dir: Path,
    optimizer_step: int,
    epoch: int,
    raw_batch: Dict[str, Any],
    predictions: Sequence[Dict[str, torch.Tensor]],
    scalar_losses: Dict[str, float],
    sample_count: int,
    seed: int,
) -> Dict[str, Any]:
    if len(predictions) != len(raw_batch["sample_ids"]):
        raise RuntimeError("monitor prediction count differs from batch size")
    eligible = [
        index for index in range(len(predictions))
        if raw_batch["pose_targets"][index].shape[0] or raw_batch["seg_targets"][index].shape[0]
    ]
    if not eligible:
        eligible = list(range(len(predictions)))
    rng = random.Random(int(seed) + int(optimizer_step))
    selected = rng.sample(eligible, k=min(int(sample_count), len(eligible)))
    monitor_dir = Path(output_dir) / "monitor" / f"step-{optimizer_step:07d}"
    monitor_dir.mkdir(parents=True, exist_ok=False)
    sample_records: List[Dict[str, Any]] = []
    for batch_index in selected:
        metrics = _sample_metrics(raw_batch, predictions[batch_index], batch_index)
        file_name = f"image-{metrics['image_id']}-batch-{batch_index}.png"
        _render_sample(raw_batch, predictions[batch_index], batch_index, metrics, monitor_dir / file_name)
        sample_records.append({**metrics, "visualization": file_name})
    payload = {
        "optimizer_step": int(optimizer_step),
        "epoch": int(epoch),
        "selection_seed": int(seed) + int(optimizer_step),
        "selected_batch_indices": selected,
        "losses": {name: float(value) for name, value in scalar_losses.items()},
        "samples": sample_records,
    }
    temporary = monitor_dir / "metrics.json.tmp"
    final = monitor_dir / "metrics.json"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(final)
    return {"monitor_dir": str(monitor_dir), "samples": sample_records}
