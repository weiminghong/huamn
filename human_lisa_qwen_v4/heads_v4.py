from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COCO_KEYPOINT_SIGMAS = torch.tensor(
    [
        0.026,
        0.025,
        0.025,
        0.035,
        0.035,
        0.079,
        0.079,
        0.072,
        0.072,
        0.062,
        0.062,
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
    ],
    dtype=torch.float32,
)


class _PoseLayer(nn.Module):
    def __init__(self, dim: int, heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            dim, heads, dropout=0.0, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        attended, _ = self.self_attention(
            query, query, query, need_weights=False
        )
        query = self.norm1(query + attended)
        attended, _ = self.cross_attention(
            query, memory, memory, need_weights=False
        )
        query = self.norm2(query + attended)
        return self.norm3(query + self.ffn(query))


class BoxFreePoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        *,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        self.state_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.visual_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.position_project = nn.Sequential(
            nn.Linear(2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.joint_embedding = nn.Parameter(
            torch.randn(17, decoder_dim) * 0.02
        )
        self.layers = nn.ModuleList(
            [_PoseLayer(decoder_dim, num_heads) for _ in range(num_layers)]
        )
        self.coordinate_head = nn.Linear(decoder_dim, 2)
        self.visibility_head = nn.Linear(decoder_dim, 3)

    def forward(
        self,
        pose_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        spatial_shape: Tuple[int, int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        persons = int(pose_states.shape[0])
        if persons == 0:
            return (
                pose_states.new_zeros((0, 17, 2)),
                pose_states.new_zeros((0, 17, 3)),
            )
        if visual_tokens.ndim != 2:
            raise ValueError("visual tokens must have shape HWxC")
        height, width = [int(value) for value in spatial_shape]
        if visual_tokens.shape[0] != height * width:
            raise ValueError("visual token count does not match spatial shape")

        y = (
            torch.arange(
                height, device=visual_tokens.device, dtype=visual_tokens.dtype
            )
            + 0.5
        ) / float(height)
        x = (
            torch.arange(
                width, device=visual_tokens.device, dtype=visual_tokens.dtype
            )
            + 0.5
        ) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        positions = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        memory = self.visual_project(visual_tokens)
        memory = memory + self.position_project(positions)
        memory = memory.unsqueeze(0).expand(persons, -1, -1)
        context = self.state_project(pose_states)
        query = context[:, None] + self.joint_embedding[None].to(context)
        for layer in self.layers:
            query = layer(query, memory)
        return torch.sigmoid(self.coordinate_head(query)), self.visibility_head(
            query
        )


class PersonMaskDecoderV4(nn.Module):
    def __init__(self, hidden_size: int, *, decoder_dim: int = 256) -> None:
        super().__init__()
        groups = 8 if decoder_dim % 8 == 0 else 1
        self.state_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.visual_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.GroupNorm(groups, decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
        )
        self.logit_scale = decoder_dim**-0.5

    def forward(
        self,
        seg_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        height, width = [int(value) for value in spatial_shape]
        if visual_tokens.shape != (height * width, visual_tokens.shape[-1]):
            raise ValueError("visual token count does not match spatial shape")
        pixels = self.visual_project(visual_tokens)
        pixels = pixels.transpose(0, 1).reshape(1, -1, height, width)
        pixels = F.interpolate(
            pixels, scale_factor=4.0, mode="bilinear", align_corners=False
        )
        pixels = self.pixel_decoder(pixels)[0]
        if seg_states.shape[0] == 0:
            return pixels.new_zeros((0, pixels.shape[-2], pixels.shape[-1]))
        masks = self.state_project(seg_states)
        return torch.einsum("nd,dhw->nhw", masks, pixels) * self.logit_scale
