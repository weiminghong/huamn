from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from human_lisa_qwen_v2.heads_v2 import (
    MultiScaleVisualNeck,
    _PoseDecoderLayer,
    inverse_sigmoid,
    mask_bce_dice_loss,
    oks_loss,
    position_encoding_2d,
    require_exact_token_states,
)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    size = boxes[..., 2:] - boxes[..., :2]
    return torch.cat([center, size], dim=-1)


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    half = 0.5 * boxes[..., 2:]
    return torch.cat([boxes[..., :2] - half, boxes[..., :2] + half], dim=-1)


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_top_left = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_bottom_right = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (enclosing_bottom_right - enclosing_top_left).clamp(min=0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-7)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "sum",
) -> torch.Tensor:
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = probability * targets + (1.0 - probability) * (1.0 - targets)
    loss = ce * (1.0 - p_t).pow(gamma)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = alpha_t * loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.mean()
    return loss


class _PersonQueryDecoderLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, query: torch.Tensor, query_position: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(query)
        positioned = normalized + query_position
        query = query + self.self_attention(positioned, positioned, normalized, need_weights=False)[0]
        normalized = self.norm2(query)
        query = query + self.cross_attention(
            normalized + query_position,
            memory,
            memory,
            need_weights=False,
        )[0]
        return query + self.ffn(self.norm3(query))


class SetPersonQueryDecoder(nn.Module):
    """DETR-style image-conditioned person set with fixed K capacity."""

    def __init__(
        self,
        *,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        num_queries: int = 25,
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()
        self.num_queries = int(num_queries)
        self.query_content = nn.Embedding(self.num_queries, decoder_dim)
        self.query_position = nn.Embedding(self.num_queries, decoder_dim)
        self.level_embedding = nn.Parameter(torch.randn(num_feature_levels, decoder_dim) * 0.02)
        self.layers = nn.ModuleList([
            _PersonQueryDecoderLayer(decoder_dim, num_heads) for _ in range(int(num_layers))
        ])
        self.objectness_heads = nn.ModuleList([nn.Linear(decoder_dim, 1) for _ in self.layers])
        self.box_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(decoder_dim),
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, 4),
            )
            for _ in self.layers
        ])
        prior_probability = 0.01
        objectness_bias = -math.log((1.0 - prior_probability) / prior_probability)
        for objectness in self.objectness_heads:
            nn.init.constant_(objectness.bias, objectness_bias)
        for box_head in self.box_heads:
            nn.init.zeros_(box_head[-1].weight)
            nn.init.zeros_(box_head[-1].bias)
            with torch.no_grad():
                box_head[-1].bias[2:].fill_(-1.5)

    def _memory(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(feature_maps) != len(self.level_embedding):
            raise ValueError("person-query feature-level count mismatch")
        pieces = []
        for level, feature in enumerate(feature_maps):
            if feature.ndim != 4 or feature.shape[0] != 1:
                raise ValueError("person-query feature maps must be 1xCxHxW")
            height, width = feature.shape[-2:]
            tokens = feature.flatten(2).transpose(1, 2)
            position = position_encoding_2d(
                height,
                width,
                tokens.shape[-1],
                device=tokens.device,
                dtype=tokens.dtype,
            ).unsqueeze(0)
            pieces.append(tokens + position + self.level_embedding[level].to(tokens))
        return torch.cat(pieces, dim=1)

    def forward(
        self,
        feature_maps: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, torch.Tensor]]]:
        memory = self._memory(feature_maps)
        query = self.query_content.weight.unsqueeze(0).to(memory)
        query_position = self.query_position.weight.unsqueeze(0).to(memory)
        layer_outputs: List[Dict[str, torch.Tensor]] = []
        for layer, objectness_head, box_head in zip(self.layers, self.objectness_heads, self.box_heads):
            query = layer(query, query_position, memory)
            objectness = objectness_head(query)[0, :, 0]
            boxes_cxcywh = box_head(query)[0].sigmoid()
            layer_outputs.append(
                {
                    "query_features": query[0],
                    "objectness_logits": objectness,
                    "boxes_cxcywh": boxes_cxcywh,
                }
            )
        final = layer_outputs[-1]
        return (
            final["query_features"],
            final["objectness_logits"],
            final["boxes_cxcywh"],
            layer_outputs[:-1],
        )


def match_person_queries(
    objectness_logits: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    *,
    class_cost_weight: float = 1.0,
    bbox_cost_weight: float = 5.0,
    giou_cost_weight: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    query_count = int(boxes_cxcywh.shape[0])
    target_count = int(target_boxes_xyxy.shape[0])
    if objectness_logits.shape != (query_count,) or boxes_cxcywh.shape != (query_count, 4):
        raise ValueError("invalid person-query prediction shapes")
    if target_boxes_xyxy.shape != (target_count, 4):
        raise ValueError("invalid person-query target box shape")
    if target_count == 0:
        empty = torch.empty(0, device=boxes_cxcywh.device, dtype=torch.long)
        return empty, empty
    if target_count > query_count:
        raise ValueError("ground-truth person count exceeds query capacity")
    probability = objectness_logits.sigmoid()
    alpha, gamma = 0.25, 2.0
    negative = (1.0 - alpha) * probability.pow(gamma) * (-(1.0 - probability + 1e-8).log())
    positive = alpha * (1.0 - probability).pow(gamma) * (-(probability + 1e-8).log())
    class_cost = (positive - negative)[:, None].expand(-1, target_count)
    target_cxcywh = box_xyxy_to_cxcywh(target_boxes_xyxy)
    bbox_cost = torch.cdist(boxes_cxcywh.float(), target_cxcywh.float(), p=1)
    giou_cost = -generalized_box_iou(
        box_cxcywh_to_xyxy(boxes_cxcywh.float()),
        target_boxes_xyxy.float(),
    )
    cost = class_cost_weight * class_cost + bbox_cost_weight * bbox_cost + giou_cost_weight * giou_cost
    query_indices, target_indices = linear_sum_assignment(cost.detach().cpu().numpy())
    return (
        torch.as_tensor(query_indices, device=boxes_cxcywh.device, dtype=torch.long),
        torch.as_tensor(target_indices, device=boxes_cxcywh.device, dtype=torch.long),
    )


def person_query_losses(
    objectness_logits: torch.Tensor,
    boxes_cxcywh: torch.Tensor,
    target_boxes_xyxy: torch.Tensor,
    query_indices: torch.Tensor,
    target_indices: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    query_count = int(objectness_logits.shape[0])
    target_count = int(target_boxes_xyxy.shape[0])
    objectness_target = objectness_logits.new_zeros(query_count)
    objectness_target[query_indices] = 1.0
    normalization = max(1, target_count)
    objectness = sigmoid_focal_loss(objectness_logits, objectness_target, reduction="sum") / normalization
    if target_count == 0:
        zero = boxes_cxcywh.sum() * 0
        return {"query_objectness": objectness, "query_bbox_l1": zero, "query_giou": zero}
    matched_prediction = boxes_cxcywh.index_select(0, query_indices)
    matched_target_xyxy = target_boxes_xyxy.index_select(0, target_indices)
    matched_target_cxcywh = box_xyxy_to_cxcywh(matched_target_xyxy)
    bbox_l1 = F.l1_loss(matched_prediction, matched_target_cxcywh, reduction="sum") / normalization
    giou = generalized_box_iou(
        box_cxcywh_to_xyxy(matched_prediction),
        matched_target_xyxy,
    )
    giou_loss = (1.0 - torch.diagonal(giou)).sum() / normalization
    return {
        "query_objectness": objectness,
        "query_bbox_l1": bbox_l1,
        "query_giou": giou_loss,
    }


class QueryConditionedPoseDecoder(nn.Module):
    """Pose queries conditioned on a matched person query and generated pose token."""

    def __init__(
        self,
        text_hidden_size: int,
        *,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        num_feature_levels: int = 3,
    ) -> None:
        super().__init__()
        self.person_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.pose_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.query_projection = nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim))
        self.joint_embedding = nn.Parameter(torch.randn(17, decoder_dim) * 0.02)
        self.level_embedding = nn.Parameter(torch.randn(num_feature_levels, decoder_dim) * 0.02)
        self.offset_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 2),
        )
        self.reference_projection = nn.Linear(2, decoder_dim)
        self.reference_feature_norm = nn.LayerNorm(decoder_dim)
        self.reference_feature_projection = nn.Linear(decoder_dim, decoder_dim)
        self.layers = nn.ModuleList([_PoseDecoderLayer(decoder_dim, num_heads) for _ in range(int(num_layers))])
        self.coordinate_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(decoder_dim),
                nn.Linear(decoder_dim, decoder_dim),
                nn.GELU(),
                nn.Linear(decoder_dim, 2),
            )
            for _ in self.layers
        ])
        self.visibility_head = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 3),
        )
        nn.init.zeros_(self.offset_head[-1].weight)
        nn.init.zeros_(self.offset_head[-1].bias)
        for head in self.coordinate_heads:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)

    def _memory(self, feature_maps: Sequence[torch.Tensor]) -> torch.Tensor:
        pieces = []
        for level, feature in enumerate(feature_maps):
            height, width = feature.shape[-2:]
            tokens = feature.flatten(2).transpose(1, 2)[0]
            position = position_encoding_2d(
                height,
                width,
                tokens.shape[-1],
                device=tokens.device,
                dtype=tokens.dtype,
            )
            pieces.append(tokens + position + self.level_embedding[level].to(tokens))
        return torch.cat(pieces, dim=0)

    def _sample(self, feature_maps: Sequence[torch.Tensor], reference: torch.Tensor) -> torch.Tensor:
        people, joints = reference.shape[:2]
        grid = reference.mul(2.0).sub(1.0).reshape(people, joints, 1, 2)
        samples = []
        for feature in feature_maps:
            sampled = F.grid_sample(
                feature.expand(people, -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )
            samples.append(sampled[..., 0].transpose(1, 2))
        value = torch.stack(samples, dim=0).mean(dim=0)
        return self.reference_feature_projection(self.reference_feature_norm(value))

    def forward(
        self,
        query_features: torch.Tensor,
        person_states: torch.Tensor,
        pose_states: torch.Tensor,
        matched_boxes_cxcywh: torch.Tensor,
        feature_maps: Sequence[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        people = int(query_features.shape[0])
        if people == 0:
            empty_xy = query_features.new_zeros((0, 17, 2))
            return empty_xy, query_features.new_zeros((0, 17, 3)), []
        context = (
            self.query_projection(query_features)
            + self.person_projection(person_states)
            + self.pose_projection(pose_states)
        )
        query = context[:, None, :] + self.joint_embedding[None].to(context)
        center = matched_boxes_cxcywh[:, None, :2]
        half_size = 0.5 * matched_boxes_cxcywh[:, None, 2:].clamp(min=1e-3)
        reference = (center + half_size * torch.tanh(self.offset_head(query))).clamp(1e-4, 1.0 - 1e-4)
        query = query + self.reference_projection(reference)
        memory = self._memory(feature_maps)
        outputs = []
        for layer, coordinate_head in zip(self.layers, self.coordinate_heads):
            query = query + self._sample(feature_maps, reference)
            query = layer(query, memory)
            reference = torch.sigmoid(inverse_sigmoid(reference) + coordinate_head(query))
            outputs.append(reference)
        return reference, self.visibility_head(query), outputs[:-1]


class QueryConditionedMaskDecoder(nn.Module):
    def __init__(self, text_hidden_size: int, *, decoder_dim: int = 256, num_heads: int = 8) -> None:
        super().__init__()
        self.query_projection = nn.Sequential(nn.LayerNorm(decoder_dim), nn.Linear(decoder_dim, decoder_dim))
        self.person_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.seg_projection = nn.Sequential(nn.LayerNorm(text_hidden_size), nn.Linear(text_hidden_size, decoder_dim))
        self.query_norm = nn.LayerNorm(decoder_dim)
        self.cross_image = nn.MultiheadAttention(decoder_dim, num_heads, batch_first=True)
        self.mask_embedding = nn.Sequential(
            nn.LayerNorm(decoder_dim),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.logit_scale = float(decoder_dim) ** -0.5

    def forward(
        self,
        query_features: torch.Tensor,
        person_states: torch.Tensor,
        seg_states: torch.Tensor,
        pose_level0: torch.Tensor,
        mask_features: torch.Tensor,
    ) -> torch.Tensor:
        people = int(query_features.shape[0])
        if people == 0:
            return mask_features.new_zeros((0, mask_features.shape[-2], mask_features.shape[-1]))
        query = (
            self.query_projection(query_features)
            + self.person_projection(person_states)
            + self.seg_projection(seg_states)
        )
        memory = pose_level0.flatten(2).transpose(1, 2).expand(people, -1, -1)
        query = query + self.cross_image(
            self.query_norm(query).unsqueeze(1),
            memory,
            memory,
            need_weights=False,
        )[0][:, 0]
        return torch.einsum("nd,dhw->nhw", self.mask_embedding(query), mask_features[0]) * self.logit_scale
