from __future__ import annotations

import torch
from torch import nn


class BoxHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 4))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        return self.layers(queries).sigmoid()


class PoseHead(nn.Module):
    def __init__(self, hidden_dim: int, num_keypoints: int = 17) -> None:
        super().__init__()
        self.num_keypoints = num_keypoints
        self.layers = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_keypoints * 3))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        batch_size, num_queries, _ = queries.shape
        return self.layers(queries).view(batch_size, num_queries, self.num_keypoints, 3)


class AttributeHead(nn.Module):
    def __init__(self, hidden_dim: int, num_gender_classes: int = 3) -> None:
        super().__init__()
        self.gender = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, num_gender_classes))
        self.age = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1))

    def forward(self, queries: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"gender_logits": self.gender(queries), "age": self.age(queries).squeeze(-1)}


class SegmentationHead(nn.Module):
    def __init__(self, hidden_dim: int, mask_channels: int = 128) -> None:
        super().__init__()
        self.mask_embed = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, mask_channels),
        )
        self.mask_feature = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, mask_channels, kernel_size=1),
        )

    def forward(self, queries: torch.Tensor, feature_map: torch.Tensor) -> torch.Tensor:
        mask_embed = self.mask_embed(queries)
        mask_feature = self.mask_feature(feature_map)
        return torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)


class PlaceholderRelationHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.pair_score = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, 1))

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        left = queries.unsqueeze(2).expand(-1, -1, queries.shape[1], -1)
        right = queries.unsqueeze(1).expand(-1, queries.shape[1], -1, -1)
        return self.pair_score(torch.cat([left, right], dim=-1)).squeeze(-1)
