from __future__ import annotations

import torch
from torch import nn

from ..schema import HumanSample, PersonSlot


class HumanMemoryProjector(nn.Module):
    def __init__(self, dense_dim: int, memory_dim: int, memory_tokens: int) -> None:
        super().__init__()
        self.memory_tokens = memory_tokens
        self.project = nn.Sequential(
            nn.LayerNorm(dense_dim),
            nn.Linear(dense_dim, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
        )

    def forward(self, dense_queries: torch.Tensor, validity_logits: torch.Tensor | None = None) -> torch.Tensor:
        memory = self.project(dense_queries)
        if validity_logits is not None:
            memory = memory * validity_logits.sigmoid().unsqueeze(-1)
        return memory[:, : self.memory_tokens, :]


def summarize_person_slot(slot: PersonSlot) -> str:
    box = "unknown"
    if slot.box_xyxy:
        box = "[" + ", ".join(str(int(round(value))) for value in slot.box_xyxy) + "]"
    pose_visible = slot.visible_pose_count()
    gender = slot.attributes.get("gender", -1)
    age = slot.attributes.get("age", -1)
    return (
        "<person_memory>"
        f"<box>{box}</box>"
        f"<pose_visible>{pose_visible}</pose_visible>"
        f"<gender>{gender}</gender>"
        f"<age>{age}</age>"
        "</person_memory>"
    )


def build_human_memory_text(sample: HumanSample) -> str:
    people = "".join(summarize_person_slot(slot) for slot in sample.slots)
    return f"<human_memory sample_id=\"{sample.sample_id}\">{people}</human_memory>"
