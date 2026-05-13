from __future__ import annotations

import torch
from torch import nn


class HumanSlotRouter(nn.Module):
    def __init__(self, hidden_dim: int, max_queries: int) -> None:
        super().__init__()
        self.max_queries = max_queries
        self.query_embed = nn.Embedding(max_queries, hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.validity = nn.Linear(hidden_dim, 1)

    def forward(self, dense_features: torch.Tensor) -> dict[str, torch.Tensor]:
        batch_size = dense_features.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(batch_size, -1, -1)
        routed, attn = self.cross_attn(queries, dense_features, dense_features)
        routed = self.norm(routed + queries)
        return {
            "queries": routed,
            "validity_logits": self.validity(routed).squeeze(-1),
            "attention": attn,
        }
