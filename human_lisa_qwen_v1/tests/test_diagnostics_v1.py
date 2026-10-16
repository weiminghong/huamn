import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.diagnostics_v1 import (
    teacher_forced_token_metrics,
)


def test_teacher_forced_metrics_use_causal_shift_and_track_tokens():
    labels = torch.tensor([[-100, 1, 2, 3]])
    logits = torch.full((1, 4, 5), -10.0)
    logits[0, 0, 1] = 10.0
    logits[0, 1, 2] = 10.0
    logits[0, 2, 0] = 10.0
    metrics = teacher_forced_token_metrics(
        logits,
        labels,
        tracked_token_ids={"end": [3], "middle": [2]},
    )
    assert metrics["supervised_tokens"] == 3
    assert metrics["correct_tokens"] == 2
    assert metrics["token_accuracy"] == pytest.approx(2 / 3)
    assert metrics["exact_sequence"] is False
    assert metrics["tracked"]["end"] == {
        "targets": 1,
        "correct": 0,
        "accuracy": 0.0,
    }
    assert metrics["tracked"]["middle"]["accuracy"] == 1.0


def test_teacher_forced_metrics_handles_no_supervised_tokens():
    metrics = teacher_forced_token_metrics(
        torch.zeros((1, 2, 4)),
        torch.full((1, 2), -100),
        tracked_token_ids={},
    )
    assert metrics["supervised_tokens"] == 0
    assert metrics["token_accuracy"] == 0.0
    assert metrics["exact_sequence"] is False
