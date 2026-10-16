from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


COCO_KEYPOINT_SIGMAS = torch.tensor(
    [
        0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
        0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089,
    ],
    dtype=torch.float32,
)


def inverse_sigmoid(value: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    return torch.logit(value.clamp(eps, 1.0 - eps))


def position_encoding_2d(height: int, width: int, dim: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 4:
        raise ValueError("2D positional encoding dimension must be divisible by 4")
    y = (torch.arange(height, device=device, dtype=torch.float32) + 0.5) / float(height)
    x = (torch.arange(width, device=device, dtype=torch.float32) + 0.5) / float(width)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    quarter = dim // 4
    omega = torch.arange(quarter, device=device, dtype=torch.float32)
    omega = 1.0 / (10000.0 ** (omega / max(1, quarter - 1)))
    xp = grid_x.reshape(-1, 1) * omega.reshape(1, -1) * (2.0 * math.pi)
    yp = grid_y.reshape(-1, 1) * omega.reshape(1, -1) * (2.0 * math.pi)
    return torch.cat([xp.sin(), xp.cos(), yp.sin(), yp.cos()], dim=-1).to(dtype=dtype)


class MultiScaleVisualNeck(nn.Module):
    """Fuse Qwen vision depths into a pose pyramid and stride-4 mask map."""

    def __init__(self, vision_hidden_size: int, num_input_layers: int, decoder_dim: int = 256) -> None:
        super().__init__()
        input_dim = int(vision_hidden_size) * int(num_input_layers)
        groups = 8 if decoder_dim % 8 == 0 else 1
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Linear(input_dim, decoder_dim)
        self.level0 = nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1), nn.GroupNorm(groups, decoder_dim), nn.GELU())
        self.level1 = nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1), nn.GroupNorm(groups, decoder_dim), nn.GELU())
        self.level2 = nn.Sequential(nn.Conv2d(decoder_dim, decoder_dim, 3, stride=2, padding=1), nn.GroupNorm(groups, decoder_dim), nn.GELU())
        self.mask_upsample = nn.Sequential(
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2), nn.GroupNorm(groups, decoder_dim), nn.GELU(),
            nn.ConvTranspose2d(decoder_dim, decoder_dim, 2, stride=2), nn.GroupNorm(groups, decoder_dim), nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
        )

    def forward(self, layer_tokens: Sequence[torch.Tensor], *, raster_indices: torch.Tensor, spatial_shape: Tuple[int, int]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if not layer_tokens:
            raise ValueError("visual neck requires at least one vision layer")
        token_count = int(layer_tokens[0].shape[0])
        if any(int(value.shape[0]) != token_count for value in layer_tokens):
            raise ValueError("selected vision layers have unequal token counts")
        height, width = [int(value) for value in spatial_shape]
        if token_count != height * width or raster_indices.numel() != token_count:
            raise ValueError("vision token count does not match raw patch grid")
        fused = self.input_projection(self.input_norm(torch.cat(layer_tokens, dim=-1)))
        raster = fused.new_zeros((height * width, fused.shape[-1])).index_copy(0, raster_indices.to(fused.device), fused)
        level0 = self.level0(raster.transpose(0, 1).reshape(1, -1, height, width))
        level1 = self.level1(level0)
        level2 = self.level2(level1)
        return [level0, level1, level2], self.mask_upsample(level0)


class _PoseDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.within_person = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.same_joint_across_people = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_image = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1, self.norm2, self.norm3, self.norm4 = [nn.LayerNorm(dim) for _ in range(4)]
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim * 4, dim))

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        people = int(query.shape[0])
        value = self.norm1(query)
        query = query + self.within_person(value, value, value, need_weights=False)[0]
        across = self.norm2(query).transpose(0, 1)
        across = self.same_joint_across_people(across, across, across, need_weights=False)[0]
        query = query + across.transpose(0, 1)
        normalized = self.norm3(query)
        image_memory = memory.unsqueeze(0).expand(people, -1, -1)
        query = query + self.cross_image(normalized, image_memory, image_memory, need_weights=False)[0]
        return query + self.ffn(self.norm4(query))


class BoxFreeJointQueryPoseDecoder(nn.Module):
    """Direct COCO-17 decoding with a learned visual person reference, never a GT box input."""

    def __init__(
        self,
        text_hidden_size: int,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_feature_levels: int = 3,
        max_people: int = 25,
        use_person_rank_queries: bool = True,
        use_visual_person_reference: bool = False,
        use_reference_feature_sampling: bool = False,
    ) -> None:
        super().__init__()
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        self.num_layers = int(num_layers)
        self.use_visual_person_reference = bool(use_visual_person_reference)
        self.use_reference_feature_sampling = bool(use_reference_feature_sampling)
        if self.use_reference_feature_sampling and not self.use_visual_person_reference:
            raise ValueError("reference feature sampling requires visual person references")
        self.person_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.pose_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.joint_embedding = nn.Parameter(torch.randn(17, decoder_dim) * 0.02)
        self.person_rank_embedding = nn.Embedding(max_people, decoder_dim) if use_person_rank_queries else None
        if self.person_rank_embedding is not None:
            nn.init.normal_(self.person_rank_embedding.weight, std=0.02)
        self.level_embedding = nn.Parameter(torch.randn(num_feature_levels, decoder_dim) * 0.02)
        if self.use_visual_person_reference:
            self.center_query_projection = nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim))
            self.center_key_norm = nn.LayerNorm(decoder_dim)
            self.center_logit_scale = float(decoder_dim) ** -0.5
        if self.use_reference_feature_sampling:
            self.reference_feature_norm = nn.LayerNorm(decoder_dim)
            self.reference_feature_projection = nn.Linear(decoder_dim, decoder_dim)
        self.reference_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )
        self.reference_projection = nn.Linear(2, decoder_dim)
        self.layers = nn.ModuleList([_PoseDecoderLayer(decoder_dim, num_heads) for _ in range(self.num_layers)])
        self.coordinate_heads = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, 2))
            for _ in range(self.num_layers)
        ])
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 3),
        )
        if self.use_visual_person_reference:
            nn.init.zeros_(self.reference_head[-1].weight)
            nn.init.zeros_(self.reference_head[-1].bias)
        for head in self.coordinate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _memory(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != len(self.level_embedding):
            raise ValueError("pose feature-level count mismatch")
        pieces = []
        for level, feature in enumerate(feature_maps):
            if feature.ndim != 4 or feature.shape[0] != 1:
                raise ValueError("pose feature maps must be 1xCxHxW")
            height, width = feature.shape[-2:]
            tokens = feature.flatten(2).transpose(1, 2)[0]
            position = position_encoding_2d(height, width, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
            pieces.append(tokens + position + self.level_embedding[level].to(tokens))
        return torch.cat(pieces, dim=0)

    def _visual_person_center(self, context: torch.Tensor, finest_feature: torch.Tensor) -> torch.Tensor:
        if not self.use_visual_person_reference:
            raise RuntimeError("visual person reference is disabled")
        if finest_feature.ndim != 4 or finest_feature.shape[0] != 1:
            raise ValueError("finest pose feature map must be 1xCxHxW")
        height, width = finest_feature.shape[-2:]
        tokens = finest_feature.flatten(2).transpose(1, 2)[0]
        position = position_encoding_2d(height, width, tokens.shape[-1], device=tokens.device, dtype=tokens.dtype)
        keys = self.center_key_norm(tokens + position)
        queries = self.center_query_projection(context)
        logits = torch.matmul(queries.float(), keys.float().transpose(0, 1)) * self.center_logit_scale
        weights = logits.softmax(dim=-1)
        y = (torch.arange(height, device=tokens.device, dtype=torch.float32) + 0.5) / float(height)
        x = (torch.arange(width, device=tokens.device, dtype=torch.float32) + 0.5) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        return torch.matmul(weights, coordinates).to(dtype=context.dtype)

    def _sample_reference_features(
        self,
        feature_maps: Sequence[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        if not self.use_reference_feature_sampling:
            raise RuntimeError("reference feature sampling is disabled")
        people, joints = reference.shape[:2]
        grid = reference.mul(2.0).sub(1.0).reshape(people, joints, 1, 2)
        sampled_levels = []
        for feature in feature_maps:
            expanded = feature.expand(people, -1, -1, -1)
            sampled = F.grid_sample(
                expanded,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            sampled_levels.append(sampled[..., 0].transpose(1, 2))
        sampled = torch.stack(sampled_levels, dim=0).mean(dim=0)
        return self.reference_feature_projection(self.reference_feature_norm(sampled))

    def forward(
        self,
        person_states: torch.Tensor,
        pose_states: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
        *,
        person_indices: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], torch.Tensor]:
        people = int(person_states.shape[0])
        if pose_states.shape != person_states.shape:
            raise ValueError("person and pose states must have equal shapes")
        if people == 0:
            empty_xy = person_states.new_zeros((0, 17, 2))
            return empty_xy, person_states.new_zeros((0, 17, 3)), [], person_states.new_zeros((0, 2))
        context = self.person_projection(person_states) + self.pose_projection(pose_states)
        if self.person_rank_embedding is not None:
            if person_indices is None:
                person_indices = torch.arange(people, device=context.device)
            person_indices = person_indices.to(device=context.device, dtype=torch.long)
            if person_indices.numel() != people or int(person_indices.max()) >= self.person_rank_embedding.num_embeddings:
                raise ValueError("pose person-rank indices exceed the configured query bank")
            context = context + self.person_rank_embedding(person_indices)
        query = context[:, None, :] + self.joint_embedding[None].to(context)
        if self.use_visual_person_reference:
            person_center = self._visual_person_center(context, feature_maps[0])
            joint_offset = 0.5 * torch.tanh(self.reference_head(query))
            reference = (person_center[:, None, :] + joint_offset).clamp(1e-4, 1.0 - 1e-4)
        else:
            reference = torch.sigmoid(self.reference_head(query))
            person_center = reference.mean(dim=1)
        query = query + self.reference_projection(reference)
        memory = self._memory(feature_maps)
        outputs: List[torch.Tensor] = []
        for layer, coordinate_head in zip(self.layers, self.coordinate_heads):
            if self.use_reference_feature_sampling:
                query = query + self._sample_reference_features(feature_maps, reference)
            query = layer(query, memory)
            reference = torch.sigmoid(inverse_sigmoid(reference) + coordinate_head(query))
            outputs.append(reference)
        return reference, self.visibility_head(query), outputs[:-1], person_center


class MultiScalePersonMaskDecoder(nn.Module):
    def __init__(self, text_hidden_size: int, decoder_dim: int = 256, num_heads: int = 8, max_people: int = 25, use_person_rank_queries: bool = True) -> None:
        super().__init__()
        self.person_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.seg_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.person_rank_embedding = nn.Embedding(max_people, decoder_dim) if use_person_rank_queries else None
        if self.person_rank_embedding is not None:
            nn.init.normal_(self.person_rank_embedding.weight, std=0.02)
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.cross_image = nn.MultiheadAttention(decoder_dim, num_heads, batch_first=True)
        self.mask_embedding = nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim), nn.GELU(), nn.Linear(decoder_dim, decoder_dim))
        self.logit_scale = float(decoder_dim) ** -0.5

    def forward(self, person_states: torch.Tensor, seg_states: torch.Tensor, pose_level0: torch.Tensor, mask_features: torch.Tensor, *, person_indices: torch.Tensor | None = None) -> torch.Tensor:
        people = int(person_states.shape[0])
        if seg_states.shape != person_states.shape:
            raise ValueError("person and seg states must have equal shapes")
        if mask_features.ndim != 4 or mask_features.shape[0] != 1:
            raise ValueError("mask feature map must be 1xCxHxW")
        if people == 0:
            return mask_features.new_zeros((0, mask_features.shape[-2], mask_features.shape[-1]))
        query = self.person_projection(person_states) + self.seg_projection(seg_states)
        if self.person_rank_embedding is not None:
            if person_indices is None:
                person_indices = torch.arange(people, device=query.device)
            person_indices = person_indices.to(device=query.device, dtype=torch.long)
            if person_indices.numel() != people or int(person_indices.max()) >= self.person_rank_embedding.num_embeddings:
                raise ValueError("mask person-rank indices exceed the configured query bank")
            query = query + self.person_rank_embedding(person_indices)
        memory = pose_level0.flatten(2).transpose(1, 2).expand(people, -1, -1)
        normalized = self.query_norm(query).unsqueeze(1)
        query = query + self.cross_image(normalized, memory, memory, need_weights=False)[0][:, 0]
        return torch.einsum("nd,dhw->nhw", self.mask_embedding(query), mask_features[0]) * self.logit_scale


def oks_loss(prediction: torch.Tensor, target: torch.Tensor, visibility: torch.Tensor, area: torch.Tensor, image_sizes: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0
    sigmas = COCO_KEYPOINT_SIGMAS.to(device=prediction.device, dtype=prediction.dtype)
    image_sizes = image_sizes.to(device=prediction.device, dtype=prediction.dtype)
    scale = torch.stack([image_sizes[:, 1], image_sizes[:, 0]], dim=-1)
    squared_distance = ((prediction - target) * scale[:, None, :]).square().sum(dim=-1)
    visible = visibility.to(device=prediction.device).gt(0)
    denominator = (2.0 * area.to(device=prediction.device, dtype=prediction.dtype)[:, None] * (sigmas * 2.0).square()[None]).clamp(min=eps)
    similarity = torch.exp(-squared_distance / denominator) * visible.to(prediction.dtype)
    count = visible.sum(dim=-1); valid = count > 0
    if not valid.any():
        return prediction.sum() * 0
    return (1.0 - similarity.sum(dim=-1)[valid] / count[valid]).mean()


def mask_bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("mask predictions and targets must have equal counts")
    if logits.shape[0] == 0:
        zero = logits.sum() * 0; return zero, zero
    target = F.interpolate(targets.to(device=logits.device, dtype=logits.dtype).unsqueeze(1), size=logits.shape[-2:], mode="nearest")[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).flatten(1).sum(dim=1) + 1.0
    denominator = probability.flatten(1).sum(dim=1) + target.flatten(1).sum(dim=1) + 1.0
    return bce, (1.0 - numerator / denominator).mean()


def require_exact_token_states(*, input_ids: torch.Tensor, hidden: torch.Tensor, labels: torch.Tensor | None, token_id: int, expected_counts: Sequence[int], task_name: str, minimum_position: int | None = None) -> List[torch.Tensor]:
    result: List[torch.Tensor] = []
    for batch_index, expected in enumerate(expected_counts):
        positions = input_ids[batch_index].eq(int(token_id))
        if labels is not None:
            positions = positions & labels[batch_index].ne(-100)
        if minimum_position is not None:
            positions = positions & torch.arange(input_ids.shape[1], device=input_ids.device).ge(int(minimum_position))
        states = hidden[batch_index][positions]
        if int(states.shape[0]) != int(expected):
            raise RuntimeError(f"{task_name} token/record mismatch at batch={batch_index}: tokens={states.shape[0]} expected={int(expected)}")
        result.append(states)
    return result
