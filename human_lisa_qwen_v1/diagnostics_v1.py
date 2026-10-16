from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import torch


def teacher_forced_token_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    tracked_token_ids: Mapping[str, Sequence[int]],
) -> Dict[str, Any]:
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("logits must be BxLxV and labels must be BxL")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logit and label sequence shapes must match")
    predictions = logits[:, :-1].argmax(dim=-1)
    targets = labels[:, 1:]
    supervised = targets.ne(-100)
    correct = predictions.eq(targets) & supervised
    supervised_count = int(supervised.sum().item())
    correct_count = int(correct.sum().item())
    row_supervised = supervised.sum(dim=1)
    row_correct = correct.sum(dim=1)
    exact_rows = (row_supervised > 0) & row_correct.eq(row_supervised)
    tracked: Dict[str, Dict[str, float | int]] = {}
    for name, ids in tracked_token_ids.items():
        ids_tensor = torch.tensor(
            [int(value) for value in ids],
            device=targets.device,
            dtype=targets.dtype,
        )
        selected = torch.isin(targets, ids_tensor) & supervised
        selected_count = int(selected.sum().item())
        selected_correct = int((correct & selected).sum().item())
        tracked[name] = {
            "targets": selected_count,
            "correct": selected_correct,
            "accuracy": (
                selected_correct / selected_count if selected_count else 0.0
            ),
        }
    return {
        "supervised_tokens": supervised_count,
        "correct_tokens": correct_count,
        "token_accuracy": (
            correct_count / supervised_count if supervised_count else 0.0
        ),
        "exact_sequence": bool(exact_rows.all().item())
        if supervised_count
        else False,
        "tracked": tracked,
    }
