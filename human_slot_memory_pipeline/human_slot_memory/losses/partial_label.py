from __future__ import annotations

from typing import Dict, List

from PIL import Image, ImageDraw

import torch
import torch.nn.functional as F
from torch import nn

from ..schema import HumanBatch


def build_slot_targets(batch: HumanBatch, max_queries: int, device: torch.device) -> Dict[str, torch.Tensor]:
    batch_size = batch.batch_size
    box = torch.zeros(batch_size, max_queries, 4, device=device)
    box_mask = torch.zeros(batch_size, max_queries, dtype=torch.bool, device=device)
    pose = torch.zeros(batch_size, max_queries, 17, 3, device=device)
    pose_mask = torch.zeros(batch_size, max_queries, dtype=torch.bool, device=device)
    seg_mask = torch.zeros(batch_size, max_queries, 56, 56, device=device)
    seg_valid = torch.zeros(batch_size, max_queries, dtype=torch.bool, device=device)
    attr_mask = torch.zeros(batch_size, max_queries, dtype=torch.bool, device=device)

    for batch_index, sample in enumerate(batch.samples):
        width = max(float(sample.width), 1.0)
        height = max(float(sample.height), 1.0)
        for slot_index, slot in enumerate(sample.slots[:max_queries]):
            if slot.task_mask.box and slot.box_xyxy:
                coords = torch.tensor(slot.box_xyxy, dtype=torch.float32, device=device)
                norm = torch.tensor([width, height, width, height], dtype=torch.float32, device=device)
                box[batch_index, slot_index] = coords / norm
                box_mask[batch_index, slot_index] = True
            if slot.task_mask.pose and slot.pose17_xyv:
                pose[batch_index, slot_index] = torch.tensor(slot.pose17_xyv, dtype=torch.float32, device=device)
                pose_mask[batch_index, slot_index] = True
            if slot.task_mask.segmentation and slot.segmentation:
                seg_mask[batch_index, slot_index] = rasterize_polygons(
                    polygons=slot.segmentation,
                    width=int(width),
                    height=int(height),
                    output_size=56,
                    device=device,
                )
                seg_valid[batch_index, slot_index] = True
            if slot.task_mask.attr:
                attr_mask[batch_index, slot_index] = True

    return {
        "box": box,
        "box_mask": box_mask,
        "pose": pose,
        "pose_mask": pose_mask,
        "seg_mask": seg_mask,
        "seg_valid": seg_valid,
        "attr_mask": attr_mask,
    }


def rasterize_polygons(polygons: List[List[float]], width: int, height: int, output_size: int, device: torch.device) -> torch.Tensor:
    canvas = Image.new("L", (max(width, 1), max(height, 1)), 0)
    drawer = ImageDraw.Draw(canvas)
    for polygon in polygons:
        points = [(polygon[idx], polygon[idx + 1]) for idx in range(0, len(polygon), 2)]
        if len(points) >= 3:
            drawer.polygon(points, fill=1, outline=1)
    resized = canvas.resize((output_size, output_size), resample=Image.NEAREST)
    data = torch.tensor(list(resized.getdata()), dtype=torch.float32, device=device).reshape(output_size, output_size)
    return data


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = logits.sigmoid()
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = 2.0 * (probs * targets).sum(dim=1) + eps
    union = probs.sum(dim=1) + targets.sum(dim=1) + eps
    return 1.0 - (intersection / union).mean()


class PartialLabelLoss(nn.Module):
    def forward(self, outputs: Dict[str, torch.Tensor], batch: HumanBatch) -> Dict[str, torch.Tensor]:
        max_queries = outputs["box_xyxy_norm"].shape[1]
        targets = build_slot_targets(batch, max_queries=max_queries, device=outputs["box_xyxy_norm"].device)
        losses: Dict[str, torch.Tensor] = {}

        if targets["box_mask"].any():
            losses["loss_box"] = F.l1_loss(
                outputs["box_xyxy_norm"][targets["box_mask"]],
                targets["box"][targets["box_mask"]],
            )
        else:
            losses["loss_box"] = outputs["box_xyxy_norm"].sum() * 0.0

        if targets["pose_mask"].any():
            pred_pose = outputs["pose17_xyv_norm"][targets["pose_mask"]][..., :2]
            gt_pose = targets["pose"][targets["pose_mask"]][..., :2]
            losses["loss_pose"] = F.l1_loss(pred_pose, gt_pose)
        else:
            losses["loss_pose"] = outputs["pose17_xyv_norm"].sum() * 0.0

        if "mask_logits" in outputs and targets["seg_valid"].any():
            pred_masks = F.interpolate(outputs["mask_logits"], size=(56, 56), mode="bilinear", align_corners=False)
            pred_masks = pred_masks[targets["seg_valid"]]
            gt_masks = targets["seg_mask"][targets["seg_valid"]]
            losses["loss_seg_bce"] = F.binary_cross_entropy_with_logits(pred_masks, gt_masks)
            losses["loss_seg_dice"] = dice_loss(pred_masks, gt_masks)
        elif "mask_logits" in outputs:
            losses["loss_seg_bce"] = outputs["mask_logits"].sum() * 0.0
            losses["loss_seg_dice"] = outputs["mask_logits"].sum() * 0.0

        losses["loss_validity"] = F.binary_cross_entropy_with_logits(
            outputs["validity_logits"],
            targets["box_mask"].float(),
        )
        return losses


def weighted_sum(losses: Dict[str, torch.Tensor], weights: Dict[str, float]) -> torch.Tensor:
    total = torch.zeros((), device=next(iter(losses.values())).device)
    for name, loss in losses.items():
        total = total + float(weights.get(name, 1.0)) * loss
    return total
