from __future__ import annotations

from typing import Dict, Mapping, Tuple

import torch
import torch.nn.functional as F

from .targets_v4 import TEXT_TASK_TO_ID


DEFAULT_TEXT_WEIGHTS = {
    "structure": 0.25,
    "box": 1.0,
    "pose_route": 0.25,
    "seg_route": 0.25,
    "gender": 1.0,
    "age": 1.0,
}


def task_normalized_causal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_ids: torch.Tensor,
    *,
    task_weights: Mapping[str, float] | None = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, int]]:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must have shapes BxLxV and BxL")
    if task_ids.shape != labels.shape:
        raise ValueError("task ids must match labels")
    weights = dict(DEFAULT_TEXT_WEIGHTS)
    if task_weights:
        weights.update({name: float(value) for name, value in task_weights.items()})
    invalid = set(weights) - set(TEXT_TASK_TO_ID)
    if invalid or any(value < 0 for value in weights.values()):
        raise ValueError(f"invalid text task weights: {invalid}")

    shifted_logits = logits[:, :-1].float().contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    shifted_tasks = task_ids[:, 1:].contiguous()
    token_loss = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shifted_labels)
    valid_labels = shifted_labels.ne(-100)
    components: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}
    weighted = []
    active_weights = []
    for name, task_id in TEXT_TASK_TO_ID.items():
        mask = valid_labels & shifted_tasks.eq(task_id)
        count = int(mask.sum().item())
        counts[name] = count
        if count:
            component = token_loss[mask].mean()
            components[name] = component
            weight = float(weights[name])
            if weight > 0:
                weighted.append(component * weight)
                active_weights.append(weight)
        else:
            components[name] = logits.sum() * 0.0
    if not weighted:
        return logits.sum() * 0.0, components, counts
    return sum(weighted) / sum(active_weights), components, counts


def pose_losses(
    coordinates: torch.Tensor,
    visibility_logits: torch.Tensor,
    targets: torch.Tensor,
    areas: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    sigmas: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    if coordinates.shape != targets[..., :2].shape:
        raise ValueError("pose coordinate and target shapes differ")
    if visibility_logits.shape != (*targets.shape[:2], 3):
        raise ValueError("pose visibility logits have an invalid shape")
    persons = int(targets.shape[0])
    if persons == 0:
        zero = coordinates.sum() * 0.0
        return {"pose_l1": zero, "pose_oks": zero, "pose_visibility": zero}

    target_visibility = targets[..., 2].long().clamp(0, 2)
    visible = target_visibility.gt(0)
    per_person_l1 = []
    for person_index in range(persons):
        person_visible = visible[person_index]
        if person_visible.any():
            per_person_l1.append(
                F.l1_loss(
                    coordinates[person_index, person_visible],
                    targets[person_index, person_visible, :2],
                )
            )
    pose_l1 = (
        torch.stack(per_person_l1).mean()
        if per_person_l1
        else coordinates.sum() * 0.0
    )
    pose_visibility = F.cross_entropy(
        visibility_logits.reshape(-1, 3),
        target_visibility.reshape(-1),
    )

    sigmas = sigmas.to(device=coordinates.device, dtype=coordinates.dtype)
    scale = torch.stack([image_sizes[:, 1], image_sizes[:, 0]], dim=-1).to(
        device=coordinates.device, dtype=coordinates.dtype
    )
    squared_distance = (
        (coordinates - targets[..., :2]) * scale[:, None, :]
    ).square().sum(dim=-1)
    variances = (sigmas * 2.0).square()
    denominator = (
        2.0
        * areas.to(device=coordinates.device, dtype=coordinates.dtype)[:, None]
        * variances[None]
    ).clamp(min=1e-6)
    similarity = torch.exp(-squared_distance / denominator)
    similarity = similarity * visible.to(similarity.dtype)
    visible_count = visible.sum(dim=-1)
    valid_person = visible_count.gt(0)
    pose_oks = (
        (
            1.0
            - similarity.sum(dim=-1)[valid_person]
            / visible_count[valid_person].to(similarity.dtype)
        ).mean()
        if valid_person.any()
        else coordinates.sum() * 0.0
    )
    return {
        "pose_l1": pose_l1,
        "pose_oks": pose_oks,
        "pose_visibility": pose_visibility,
    }


def mask_losses(
    logits: torch.Tensor, targets: torch.Tensor
) -> Dict[str, torch.Tensor]:
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("mask predictions and targets have different counts")
    if logits.shape[0] == 0:
        zero = logits.sum() * 0.0
        return {"mask_bce": zero, "mask_dice": zero}
    target = targets.to(device=logits.device, dtype=logits.dtype).unsqueeze(1)
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")[:, 0]
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    mask_bce = bce_map.flatten(1).mean(dim=1).mean()
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).flatten(1).sum(dim=1) + 1.0
    denominator = (
        probability.flatten(1).sum(dim=1)
        + target.flatten(1).sum(dim=1)
        + 1.0
    )
    return {
        "mask_bce": mask_bce,
        "mask_dice": (1.0 - numerator / denominator).mean(),
    }


def active_weighted_mean(
    losses: Mapping[str, torch.Tensor],
    weights: Mapping[str, float],
    active: Mapping[str, bool],
) -> torch.Tensor:
    terms = []
    denominators = []
    for name, loss in losses.items():
        weight = float(weights.get(name, 0.0))
        if active.get(name, False) and weight > 0:
            terms.append(loss * weight)
            denominators.append(weight)
    if not terms:
        reference = next(iter(losses.values()))
        return reference.sum() * 0.0
    return sum(terms) / sum(denominators)
