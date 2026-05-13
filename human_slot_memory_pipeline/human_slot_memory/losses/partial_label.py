from __future__ import annotations

from typing import Dict

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
            if slot.task_mask.attr:
                attr_mask[batch_index, slot_index] = True

    return {
        "box": box,
        "box_mask": box_mask,
        "pose": pose,
        "pose_mask": pose_mask,
        "attr_mask": attr_mask,
    }


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
