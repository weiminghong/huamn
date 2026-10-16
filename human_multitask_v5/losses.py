from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    labels = targets.float().flatten(1)
    numerator = 2 * (probabilities * labels).sum(1)
    denominator = probabilities.sum(1) + labels.sum(1)
    return (1 - (numerator + 1) / (denominator + 1)).mean()


def segment_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if logits.shape != targets.shape:
        raise ValueError("segment logits and targets must have equal shapes")
    if logits.numel() == 0:
        zero = logits.sum() * 0
        return {"bce": zero, "dice": zero, "total": zero}
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    dice = dice_loss(logits.float(), targets.float())
    return {"bce": bce, "dice": dice, "total": bce + dice}


def parsing_loss(
    logits: torch.Tensor,
    label_map: torch.Tensor,
    *,
    ignore_index: int = 255,
) -> Dict[str, torch.Tensor]:
    """Class CE plus foreground part-mask Dice.

    logits are [N, 58, H, W] for foreground MHP parts. Background is an
    explicit zero-logit channel so it cannot become a hidden 59th query.
    """
    if logits.ndim != 4 or label_map.ndim != 3:
        raise ValueError("parsing shapes must be [N,C,H,W] and [N,H,W]")
    if logits.shape[0] != label_map.shape[0]:
        raise ValueError("parsing batch sizes differ")
    background = logits.new_zeros(
        (logits.shape[0], 1, logits.shape[2], logits.shape[3])
    )
    classes = torch.cat([background, logits], dim=1)
    ce = F.cross_entropy(
        classes.float(), label_map.long(), ignore_index=int(ignore_index)
    )
    valid = label_map.ne(ignore_index)
    target_parts = torch.stack(
        [
            (label_map == part_id) & valid
            for part_id in range(1, logits.shape[1] + 1)
        ],
        dim=1,
    )
    dice = dice_loss(
        logits.flatten(0, 1).float(),
        target_parts.flatten(0, 1).float(),
    )
    return {"ce": ce, "dice": dice, "total": ce + dice}
