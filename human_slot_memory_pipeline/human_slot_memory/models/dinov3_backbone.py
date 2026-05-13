from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class BackboneOutput:
    dense_features: torch.Tensor
    pooled: torch.Tensor
    feature_map: torch.Tensor


class DinoV3Backbone(nn.Module):
    def __init__(self, model_path: str, hidden_dim: int = 256, freeze: bool = True, dry_run: bool = False) -> None:
        super().__init__()
        self.model_path = model_path
        self.hidden_dim = hidden_dim
        self.dry_run = dry_run
        self.model: Optional[nn.Module] = None
        self.proj: Optional[nn.Module] = None
        if not dry_run:
            self._load_model(freeze=freeze)

    def _load_model(self, freeze: bool) -> None:
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(self.model_path, trust_remote_code=True)
        model_dim = int(getattr(self.model.config, "hidden_size", self.hidden_dim))
        self.proj = nn.Linear(model_dim, self.hidden_dim)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> BackboneOutput:
        if self.dry_run or self.model is None:
            batch_size = pixel_values.shape[0]
            dense = torch.zeros(batch_size, 196, self.hidden_dim, device=pixel_values.device)
            feature_map = dense.transpose(1, 2).reshape(batch_size, self.hidden_dim, 14, 14)
            return BackboneOutput(dense_features=dense, pooled=dense.mean(dim=1), feature_map=feature_map)

        outputs = self.model(pixel_values=pixel_values, output_hidden_states=True)
        tokens = outputs.last_hidden_state
        if tokens.shape[1] > 1:
            spatial_tokens = tokens[:, 1:, :]
        else:
            spatial_tokens = tokens
        if self.proj is None:
            raise RuntimeError("DINOv3 projection layer was not initialized")
        dense = self.proj(spatial_tokens)
        side = int(dense.shape[1] ** 0.5)
        feature_map = dense[:, : side * side, :].transpose(1, 2).reshape(dense.shape[0], self.hidden_dim, side, side)
        return BackboneOutput(dense_features=dense, pooled=dense.mean(dim=1), feature_map=feature_map)
