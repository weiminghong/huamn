from __future__ import annotations

import torch
from torch import nn


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
