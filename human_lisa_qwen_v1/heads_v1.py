from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

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

COCO_JOINT_REFERENCE = torch.tensor(
    [
        [0.50, 0.15], [0.46, 0.12], [0.54, 0.12],
        [0.40, 0.14], [0.60, 0.14],
        [0.36, 0.32], [0.64, 0.32],
        [0.28, 0.48], [0.72, 0.48],
        [0.22, 0.64], [0.78, 0.64],
        [0.43, 0.57], [0.57, 0.57],
        [0.40, 0.75], [0.60, 0.75],
        [0.38, 0.93], [0.62, 0.93],
    ],
    dtype=torch.float32,
 )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (
        boxes[..., 3] - boxes[..., 1]
    ).clamp(min=0)


def jitter_pose_boxes(
    boxes: torch.Tensor,
    *,
    shift_factor: float = 0.16,
    shift_prob: float = 0.3,
    scale_range: Tuple[float, float] = (0.5, 1.5),
    scale_prob: float = 1.0,
    minimum_size: float = 1e-4,
) -> torch.Tensor:
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("pose boxes must have shape Nx4")
    shift_factor = float(shift_factor)
    shift_prob = float(shift_prob)
    scale_min, scale_max = [float(value) for value in scale_range]
    scale_prob = float(scale_prob)
    minimum_size = float(minimum_size)
    if shift_factor < 0:
        raise ValueError("pose box shift factor must be non-negative")
    if not 0.0 <= shift_prob <= 1.0:
        raise ValueError("pose box shift probability must be in [0, 1]")
    if scale_min <= 0 or scale_max < scale_min:
        raise ValueError("pose box scale range must be positive and ordered")
    if not 0.0 <= scale_prob <= 1.0:
        raise ValueError("pose box scale probability must be in [0, 1]")
    if not 0.0 < minimum_size < 1.0:
        raise ValueError("pose box minimum size must be in (0, 1)")
    if boxes.numel() == 0:
        return boxes.clone()
    if (
        shift_factor == 0.0
        and scale_min == 1.0
        and scale_max == 1.0
    ):
        return boxes.clone()

    boxes_float = boxes.float()
    top_left = boxes_float[:, :2]
    size = (boxes_float[:, 2:] - top_left).clamp(min=minimum_size)
    center = top_left + size * 0.5
    persons = int(boxes.shape[0])

    if shift_factor > 0 and shift_prob > 0:
        shift_noise = torch.randn(
            (persons, 2), device=boxes.device, dtype=boxes_float.dtype
        ).clamp(-1.0, 1.0)
        shift_mask = (
            torch.rand((persons, 1), device=boxes.device) < shift_prob
        ).to(boxes_float.dtype)
        center = center + shift_noise * shift_factor * size * shift_mask

    if scale_prob > 0 and (scale_min != 1.0 or scale_max != 1.0):
        scale_noise = torch.randn(
            (persons, 1), device=boxes.device, dtype=boxes_float.dtype
        ).clamp(-1.0, 1.0)
        scale = scale_noise * ((scale_max - scale_min) * 0.5) + (
            (scale_max + scale_min) * 0.5
        )
        scale_mask = torch.rand((persons, 1), device=boxes.device) < scale_prob
        scale = torch.where(scale_mask, scale, torch.ones_like(scale))
        size = size * scale

    jittered_top_left = (center - size * 0.5).clamp(
        min=0.0, max=1.0 - minimum_size
    )
    jittered_bottom_right = (center + size * 0.5).clamp(
        min=minimum_size, max=1.0
    )
    jittered_bottom_right = torch.maximum(
        jittered_bottom_right, jittered_top_left + minimum_size
    ).clamp(max=1.0)
    return torch.cat([jittered_top_left, jittered_bottom_right], dim=-1).to(
        dtype=boxes.dtype
    )


def keypoints_inside_boxes(
    target_xy: torch.Tensor, boxes: torch.Tensor
) -> torch.Tensor:
    if target_xy.ndim != 3 or target_xy.shape[-1] != 2:
        raise ValueError("pose keypoints must have shape NxKx2")
    if boxes.shape != (target_xy.shape[0], 4):
        raise ValueError("pose boxes must have shape Nx4")
    top_left = boxes[:, None, :2].to(target_xy)
    bottom_right = boxes[:, None, 2:].to(target_xy)
    return ((target_xy >= top_left) & (target_xy <= bottom_right)).all(dim=-1)


def aligned_box_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 4:
        raise ValueError("aligned boxes must have matching Nx4 shapes")
    top_left = torch.maximum(prediction[:, :2], target[:, :2])
    bottom_right = torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = box_area(prediction) + box_area(target) - intersection
    return intersection / union.clamp(min=1e-7)


def generalized_box_iou_loss(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0
    top_left = torch.maximum(prediction[:, :2], target[:, :2])
    bottom_right = torch.minimum(prediction[:, 2:], target[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
    union = box_area(prediction) + box_area(target) - intersection
    iou = intersection / union.clamp(min=1e-7)
    enclosing_top_left = torch.minimum(prediction[:, :2], target[:, :2])
    enclosing_bottom_right = torch.maximum(prediction[:, 2:], target[:, 2:])
    enclosing_area = (enclosing_bottom_right - enclosing_top_left).clamp(
        min=0
    ).prod(dim=-1)
    giou = iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-7)
    return (1.0 - giou).mean()


class BoxQualityHead(nn.Module):
    def __init__(self, hidden_size: int, decoder_dim: int = 256) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
            nn.GELU(),
        )
        self.box_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 4),
        )
        self.quality_mlp = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 1),
        )

    def forward(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.project(states)
        cxcywh = torch.sigmoid(self.box_mlp(features))
        center = cxcywh[:, :2]
        size = cxcywh[:, 2:].clamp(min=1e-4)
        top_left = (center - size * 0.5).clamp(min=0.0, max=1.0)
        bottom_right = (center + size * 0.5).clamp(min=0.0, max=1.0)
        bottom_right = torch.maximum(bottom_right, top_left + 1e-5).clamp(
            max=1.0
        )
        boxes = torch.cat([top_left, bottom_right], dim=-1)
        quality_logits = self.quality_mlp(features).squeeze(-1)
        return boxes, quality_logits


def roi_pool_visual_tokens(
    visual_tokens: torch.Tensor,
    *,
    spatial_shape: Tuple[int, int],
    boxes: torch.Tensor,
    output_size: int = 7,
) -> torch.Tensor:
    if visual_tokens.ndim == 3:
        if visual_tokens.shape[0] != 1:
            raise ValueError("ROI pooling expects one image of visual tokens")
        visual_tokens = visual_tokens[0]
    if visual_tokens.ndim != 2:
        raise ValueError("visual tokens must be HWxC or 1xHWxC")
    height, width = [int(value) for value in spatial_shape]
    if visual_tokens.shape[0] != height * width:
        raise ValueError("visual token count does not match spatial shape")
    if boxes.ndim != 2 or boxes.shape[-1] != 4:
        raise ValueError("ROI boxes must have shape Nx4")
    output_size = int(output_size)
    if output_size <= 0:
        raise ValueError("ROI output_size must be positive")
    persons = int(boxes.shape[0])
    channels = int(visual_tokens.shape[-1])
    if persons == 0:
        return visual_tokens.new_zeros(
            (0, output_size * output_size, channels)
        )
    boxes = boxes.to(device=visual_tokens.device, dtype=visual_tokens.dtype)
    boxes = boxes.clamp(0.0, 1.0)
    top_left = boxes[:, :2]
    bottom_right = torch.maximum(boxes[:, 2:], top_left + 1e-4).clamp(
        max=1.0
    )
    fractions = (
        torch.arange(
            output_size,
            device=visual_tokens.device,
            dtype=visual_tokens.dtype,
        )
        + 0.5
    ) / float(output_size)
    xs = top_left[:, 0, None] + (
        bottom_right[:, 0, None] - top_left[:, 0, None]
    ) * fractions[None, :]
    ys = top_left[:, 1, None] + (
        bottom_right[:, 1, None] - top_left[:, 1, None]
    ) * fractions[None, :]
    grid_x = xs[:, None, :].expand(-1, output_size, -1)
    grid_y = ys[:, :, None].expand(-1, -1, output_size)
    grid = torch.stack([grid_x, grid_y], dim=-1).mul(2.0).sub(1.0)
    feature_map = visual_tokens.transpose(0, 1).reshape(
        1, channels, height, width
    )
    pooled = F.grid_sample(
        feature_map.expand(persons, -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    return pooled.flatten(2).transpose(1, 2)


def keypoint_heatmap_mse_loss(
    prediction: torch.Tensor,
    target_xy: torch.Tensor,
    visibility: torch.Tensor,
    boxes: torch.Tensor,
    *,
    sigma: float = 2.0,
    return_target: bool = False,
):
    if prediction.ndim != 4:
        raise ValueError("pose heatmaps must have shape NxKxHxW")
    if target_xy.shape != prediction.shape[:2] + (2,):
        raise ValueError("pose heatmap targets must have shape NxKx2")
    if visibility.shape != prediction.shape[:2]:
        raise ValueError("pose visibility must have shape NxK")
    if boxes.shape != (prediction.shape[0], 4):
        raise ValueError("pose boxes must have shape Nx4")
    if sigma <= 0:
        raise ValueError("pose heatmap sigma must be positive")
    height, width = prediction.shape[-2:]
    boxes = boxes.to(device=prediction.device, dtype=prediction.dtype)
    target_xy = target_xy.to(device=prediction.device, dtype=prediction.dtype)
    top_left = boxes[:, :2]
    box_size = (boxes[:, 2:] - top_left).clamp(min=1e-4)
    local = ((target_xy - top_left[:, None, :]) / box_size[:, None, :]).clamp(
        0.0, 1.0
    )
    center_x = local[..., 0] * float(max(1, width - 1))
    center_y = local[..., 1] * float(max(1, height - 1))
    grid_x = torch.arange(
        width, device=prediction.device, dtype=prediction.dtype
    ).view(1, 1, 1, width)
    grid_y = torch.arange(
        height, device=prediction.device, dtype=prediction.dtype
    ).view(1, 1, height, 1)
    squared_distance = (
        (grid_x - center_x[:, :, None, None]).square()
        + (grid_y - center_y[:, :, None, None]).square()
    )
    target = torch.exp(-squared_distance / (2.0 * float(sigma) ** 2))
    visible = visibility.to(device=prediction.device).gt(0)
    target = target * visible[:, :, None, None].to(target.dtype)
    if visible.any():
        per_joint = (prediction - target).square().mean(dim=(-1, -2))
        loss = (
            per_joint * visible.to(per_joint.dtype)
        ).sum() / visible.sum().clamp(min=1)
    else:
        loss = prediction.sum() * 0
    if return_target:
        return loss, target
    return loss


class TopDownHeatmapPoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        visual_hidden_size: int,
        decoder_dim: int = 256,
        heatmap_size: int = 56,
        softargmax_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        if heatmap_size <= 0 or heatmap_size % 8:
            raise ValueError("heatmap_size must be a positive multiple of 8")
        if softargmax_temperature <= 0:
            raise ValueError("softargmax temperature must be positive")
        self.decoder_dim = int(decoder_dim)
        self.heatmap_size = int(heatmap_size)
        self.softargmax_temperature = float(softargmax_temperature)
        self.visual_project = nn.Linear(visual_hidden_size, decoder_dim)
        self.condition = nn.Sequential(
            nn.LayerNorm(hidden_size * 2 + 4),
            nn.Linear(hidden_size * 2 + 4, decoder_dim * 2),
        )
        groups = 8 if decoder_dim % 8 == 0 else 1
        deconvolution = []
        for _ in range(3):
            deconvolution.extend(
                [
                    nn.ConvTranspose2d(
                        decoder_dim,
                        decoder_dim,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(groups, decoder_dim),
                    nn.ReLU(inplace=True),
                ]
            )
        self.deconvolution = nn.Sequential(*deconvolution)
        self.final_layer = nn.Conv2d(decoder_dim, 17, kernel_size=1)
        nn.init.normal_(self.final_layer.weight, std=0.001)
        nn.init.constant_(self.final_layer.bias, 0.0)
        self.joint_embedding = nn.Parameter(
            torch.randn(17, decoder_dim) * 0.02
        )
        self.visibility_norm = nn.LayerNorm(decoder_dim)
        self.visibility_head = nn.Linear(decoder_dim, 3)

    def forward(
        self,
        pose_states: torch.Tensor,
        parent_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        person_boxes: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        persons = int(pose_states.shape[0])
        if parent_states.shape[0] != persons:
            raise ValueError("pose and parent states must have equal person counts")
        if person_boxes.shape != (persons, 4):
            raise ValueError("person_boxes must have shape persons x 4")
        if visual_tokens.ndim != 3 or visual_tokens.shape[0] != persons:
            raise ValueError("pose visual tokens must be one ROI per person")
        if persons == 0:
            return (
                pose_states.new_zeros((0, 17, 2)),
                pose_states.new_zeros((0, 17, 3)),
                pose_states.new_zeros(
                    (0, 17, self.heatmap_size, self.heatmap_size)
                ),
            )
        token_count = int(visual_tokens.shape[1])
        side = int(round(token_count ** 0.5))
        if side * side != token_count:
            raise ValueError("pose ROI tokens must form a square feature map")
        boxes = person_boxes.to(
            device=pose_states.device, dtype=pose_states.dtype
        ).clamp(0.0, 1.0)
        condition = self.condition(
            torch.cat([pose_states, parent_states, boxes], dim=-1)
        )
        scale, shift = condition.chunk(2, dim=-1)
        feature_map = self.visual_project(visual_tokens)
        feature_map = feature_map.transpose(1, 2).reshape(
            persons, self.decoder_dim, side, side
        )
        feature_map = (
            feature_map * (1.0 + torch.tanh(scale)[:, :, None, None])
            + shift[:, :, None, None]
        )
        heatmaps = self.final_layer(self.deconvolution(feature_map))
        if heatmaps.shape[-2:] != (self.heatmap_size, self.heatmap_size):
            heatmaps = F.interpolate(
                heatmaps,
                size=(self.heatmap_size, self.heatmap_size),
                mode="bilinear",
                align_corners=False,
            )
        probabilities = (
            heatmaps.float().flatten(2) / self.softargmax_temperature
        ).softmax(dim=-1)
        axis = torch.linspace(
            0.0,
            1.0,
            self.heatmap_size,
            device=heatmaps.device,
            dtype=probabilities.dtype,
        )
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        local_x = (probabilities * grid_x.flatten()[None, None]).sum(dim=-1)
        local_y = (probabilities * grid_y.flatten()[None, None]).sum(dim=-1)
        local = torch.stack([local_x, local_y], dim=-1)
        boxes_float = boxes.float()
        top_left = boxes_float[:, :2]
        size = (boxes_float[:, 2:] - top_left).clamp(min=1e-4)
        coordinates = top_left[:, None, :] + local * size[:, None, :]
        pooled = feature_map.mean(dim=(-1, -2))
        joint_features = self.visibility_norm(
            pooled[:, None, :] + self.joint_embedding[None]
        )
        visibility = self.visibility_head(joint_features)
        return coordinates.clamp(0.0, 1.0), visibility, heatmaps


def sample_reference_features(
    feature_map: torch.Tensor,
    references: torch.Tensor,
    offsets: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    if feature_map.ndim != 4:
        raise ValueError("feature_map must have shape BxCxHxW")
    if references.ndim != 3 or references.shape[-1] != 2:
        raise ValueError("references must have shape BxJx2")
    if offsets.ndim != 4 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape BxJxPx2")
    if weights.shape != offsets.shape[:-1]:
        raise ValueError("weights must have shape BxJxP")
    if references.shape[:2] != offsets.shape[:2]:
        raise ValueError("reference and offset person/joint axes must match")
    if feature_map.shape[0] != references.shape[0]:
        raise ValueError("feature and reference batch axes must match")
    locations = references[:, :, None, :] + offsets
    grid = locations.mul(2.0).sub(1.0)
    sampled = F.grid_sample(
        feature_map,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    sampled = sampled.permute(0, 2, 3, 1)
    return (sampled * weights[..., None]).sum(dim=2)


class ReferencePointJointRefinement(nn.Module):
    def __init__(
        self,
        decoder_dim: int,
        num_heads: int,
        *,
        num_points: int = 8,
        offset_radius: float = 2.0,
    ) -> None:
        super().__init__()
        if num_points <= 0:
            raise ValueError("reference refinement requires sampling points")
        self.num_points = int(num_points)
        self.offset_radius = float(offset_radius)
        self.self_norm = nn.LayerNorm(decoder_dim)
        self.self_attention = nn.MultiheadAttention(
            decoder_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.sample_norm = nn.LayerNorm(decoder_dim)
        self.sampling_offsets = nn.Linear(decoder_dim, self.num_points * 2)
        self.sampling_weights = nn.Linear(decoder_dim, self.num_points)
        self.sample_project = nn.Linear(decoder_dim, decoder_dim)
        self.cross_norm = nn.LayerNorm(decoder_dim)
        self.ffn = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim * 4),
            nn.GELU(),
            nn.Linear(decoder_dim * 4, decoder_dim),
        )
        self.ffn_norm = nn.LayerNorm(decoder_dim)
        nn.init.zeros_(self.sampling_offsets.weight)
        angles = torch.arange(self.num_points, dtype=torch.float32)
        angles = angles * (2.0 * math.pi / float(self.num_points))
        directions = torch.stack([angles.cos(), angles.sin()], dim=-1)
        with torch.no_grad():
            self.sampling_offsets.bias.copy_(directions.flatten())
        nn.init.zeros_(self.sampling_weights.weight)
        nn.init.zeros_(self.sampling_weights.bias)

    def forward(
        self,
        query: torch.Tensor,
        feature_map: torch.Tensor,
        references: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.self_norm(query)
        attended, _ = self.self_attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        query = query + attended
        normalized = self.sample_norm(query)
        batch, joints, _ = normalized.shape
        height, width = feature_map.shape[-2:]
        offsets = self.sampling_offsets(normalized).reshape(
            batch, joints, self.num_points, 2
        )
        scale = offsets.new_tensor(
            [self.offset_radius / float(width), self.offset_radius / float(height)]
        )
        offsets = torch.tanh(offsets) * scale
        weights = self.sampling_weights(normalized).softmax(dim=-1)
        sampled = sample_reference_features(
            feature_map,
            references,
            offsets,
            weights,
        )
        query = self.cross_norm(query + self.sample_project(sampled))
        return self.ffn_norm(query + self.ffn(query))


class DirectJointQueryPoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        visual_hidden_size: int,
        decoder_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        if num_layers <= 0:
            raise ValueError("direct pose decoder requires at least one layer")
        self.decoder_dim = int(decoder_dim)
        self.num_layers = int(num_layers)
        self.pose_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.parent_project = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )
        self.box_project = nn.Sequential(
            nn.Linear(4, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.visual_project = nn.Sequential(
            nn.LayerNorm(visual_hidden_size),
            nn.Linear(visual_hidden_size, decoder_dim),
        )
        self.visual_position = nn.Sequential(
            nn.Linear(2, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
        )
        self.reference_project = nn.Linear(2, decoder_dim)
        self.joint_embedding = nn.Parameter(
            torch.randn(17, decoder_dim) * 0.02
        )
        self.joint_reference_logits = nn.Parameter(
            torch.logit(COCO_JOINT_REFERENCE.clamp(1e-4, 1.0 - 1e-4))
        )
        self.refinement_layers = nn.ModuleList(
            [
                ReferencePointJointRefinement(
                    decoder_dim,
                    num_heads,
                    num_points=8,
                    offset_radius=2.0,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.coordinate_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(decoder_dim),
                    nn.Linear(decoder_dim, decoder_dim),
                    nn.GELU(),
                    nn.Linear(decoder_dim, 2),
                )
                for _ in range(self.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(decoder_dim)
        self.visibility_head = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, 3),
        )
        for coordinate_head in self.coordinate_heads:
            nn.init.zeros_(coordinate_head[-1].weight)

    def forward(
        self,
        pose_states: torch.Tensor,
        parent_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        spatial_shape: Tuple[int, int],
        person_boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        persons = int(pose_states.shape[0])
        if parent_states.shape != pose_states.shape:
            raise ValueError("pose and parent states must have matching shapes")
        if persons == 0:
            return (
                pose_states.new_zeros((0, 17, 2)),
                pose_states.new_zeros((0, 17, 3)),
            )
        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        if visual_tokens.ndim != 3:
            raise ValueError("visual tokens must have shape BxHWxC")
        if visual_tokens.shape[0] == 1:
            visual_tokens = visual_tokens.expand(persons, -1, -1)
        elif visual_tokens.shape[0] != persons:
            raise ValueError("visual tokens must contain one image or one per person")
        height, width = [int(value) for value in spatial_shape]
        if height <= 0 or width <= 0:
            raise ValueError("visual spatial shape must be positive")
        if visual_tokens.shape[1] != height * width:
            raise ValueError("visual token count does not match spatial shape")

        reference_local = torch.sigmoid(self.joint_reference_logits).to(
            device=pose_states.device, dtype=pose_states.dtype
        )
        reference = reference_local.unsqueeze(0).expand(persons, -1, -1)
        context = self.pose_project(pose_states) + self.parent_project(
            parent_states
        )
        if person_boxes is not None:
            if person_boxes.shape != (persons, 4):
                raise ValueError("person_boxes must have shape persons x 4")
            boxes = person_boxes.to(
                device=pose_states.device, dtype=pose_states.dtype
            ).clamp(0.0, 1.0)
            top_left = boxes[:, :2]
            size = (boxes[:, 2:] - top_left).clamp(min=1e-4)
            reference = top_left[:, None, :] + reference * size[:, None, :]
            context = context + self.box_project(boxes)

        query = (
            context[:, None, :]
            + self.joint_embedding[None].to(context)
            + self.reference_project(reference)
        )
        y = (
            torch.arange(height, device=pose_states.device, dtype=pose_states.dtype)
            + 0.5
        ) / float(height)
        x = (
            torch.arange(width, device=pose_states.device, dtype=pose_states.dtype)
            + 0.5
        ) / float(width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        position = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        memory = self.visual_project(visual_tokens) + self.visual_position(
            position
        )[None]
        feature_map = memory.transpose(1, 2).reshape(
            persons, self.decoder_dim, height, width
        )
        coordinates = reference
        for layer, coordinate_head in zip(
            self.refinement_layers, self.coordinate_heads
        ):
            query = layer(query, feature_map, coordinates)
            delta = coordinate_head(query)
            coordinates = torch.sigmoid(
                torch.logit(coordinates.clamp(1e-4, 1.0 - 1e-4)) + delta
            )
        query = self.final_norm(query)
        return coordinates, self.visibility_head(query)
class MultiPersonPoseDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        visual_hidden_size: int,
        decoder_dim: int = 256,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        if decoder_dim % num_heads:
            raise ValueError("decoder_dim must be divisible by num_heads")
        self.decoder_dim = int(decoder_dim)
        self.pose_project = nn.Linear(hidden_size, decoder_dim)
        self.parent_project = nn.Linear(hidden_size, decoder_dim)
        self.box_project = nn.Linear(4, decoder_dim)
        self.visual_project = nn.Linear(visual_hidden_size, decoder_dim)
        self.joint_reference_logits = nn.Parameter(
            torch.logit(COCO_JOINT_REFERENCE.clamp(1e-4, 1.0 - 1e-4))
        )
        self.joint_embedding = nn.Parameter(torch.randn(17, decoder_dim) * 0.02)
        self.within_person = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.across_people = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.visual_cross_attention = nn.MultiheadAttention(
            decoder_dim, num_heads, dropout=0.0, batch_first=True
        )
        self.norm1 = nn.LayerNorm(decoder_dim)
        self.norm2 = nn.LayerNorm(decoder_dim)
        self.norm3 = nn.LayerNorm(decoder_dim)
        self.xy_head = nn.Linear(decoder_dim, 2)
        self.visibility_head = nn.Linear(decoder_dim, 3)

    def forward(
        self,
        pose_states: torch.Tensor,
        parent_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        use_joint_reference: bool = True,
        person_boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        persons = int(pose_states.shape[0])
        if parent_states.shape[0] != persons:
            raise ValueError("pose and parent states must have equal person counts")
        if persons == 0:
            return (
                pose_states.new_zeros((0, 17, 2)),
                pose_states.new_zeros((0, 17, 3)),
            )
        query = self.pose_project(pose_states) + self.parent_project(parent_states)
        if person_boxes is not None:
            if person_boxes.shape != (persons, 4):
                raise ValueError("person_boxes must have shape persons x 4")
            query = query + self.box_project(
                person_boxes.to(device=query.device, dtype=query.dtype)
            )
        query = query[:, None, :] + self.joint_embedding[None, :, :]
        attended, _ = self.within_person(query, query, query, need_weights=False)
        query = self.norm1(query + attended)

        by_joint = query.transpose(0, 1)
        attended, _ = self.across_people(
            by_joint, by_joint, by_joint, need_weights=False
        )
        query = self.norm2((by_joint + attended).transpose(0, 1))

        if visual_tokens.ndim == 2:
            visual_tokens = visual_tokens.unsqueeze(0)
        if visual_tokens.shape[0] == 1:
            visual_tokens = visual_tokens.expand(persons, -1, -1)
        elif visual_tokens.shape[0] != persons:
            raise ValueError(
                "pose visual tokens must be one image or one ROI per person"
            )
        visual = self.visual_project(visual_tokens)
        attended, _ = self.visual_cross_attention(
            query, visual, visual, need_weights=False
        )
        query = self.norm3(query + attended)
        coordinate_logits = self.xy_head(query)
        if use_joint_reference:
            coordinate_logits = (
                coordinate_logits + self.joint_reference_logits[None]
            )
        coordinates = torch.sigmoid(coordinate_logits)
        if person_boxes is not None:
            boxes = person_boxes.to(
                device=coordinates.device, dtype=coordinates.dtype
            ).clamp(0.0, 1.0)
            top_left = boxes[:, :2]
            size = (boxes[:, 2:] - top_left).clamp(min=1e-4)
            coordinates = top_left[:, None, :] + coordinates * size[:, None, :]
            coordinates = coordinates.clamp(0.0, 1.0)
        return coordinates, self.visibility_head(query)


def roi_relative_pose_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("pose predictions and targets must have matching NxKx2")
    if boxes.shape != (prediction.shape[0], 4):
        raise ValueError("pose boxes must have shape Nx4")
    visible = visibility.to(device=prediction.device).gt(0)
    if not visible.any():
        return prediction.sum() * 0
    boxes = boxes.to(device=prediction.device, dtype=prediction.dtype)
    box_size = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-4)
    relative_error = (prediction - target).abs() / box_size[:, None, :]
    expanded = visible.unsqueeze(-1).expand_as(relative_error)
    return relative_error[expanded].mean()


def oks_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    visibility: torch.Tensor,
    area: torch.Tensor,
    image_sizes: torch.Tensor,
    *,
    sigmas: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    if prediction.numel() == 0:
        return prediction.sum() * 0
    if prediction.shape != target.shape or prediction.shape[-1] != 2:
        raise ValueError("pose predictions and targets must have matching NxKx2")
    sigmas = COCO_KEYPOINT_SIGMAS if sigmas is None else sigmas
    sigmas = sigmas.to(device=prediction.device, dtype=prediction.dtype)
    if sigmas.numel() != prediction.shape[1]:
        raise ValueError("sigma count does not match keypoint count")
    image_sizes = image_sizes.to(device=prediction.device, dtype=prediction.dtype)
    scale = torch.stack([image_sizes[:, 1], image_sizes[:, 0]], dim=-1)
    difference = (prediction - target) * scale[:, None, :]
    squared_distance = difference.square().sum(dim=-1)
    visible = visibility.to(device=prediction.device).gt(0)
    variances = (sigmas * 2.0).square()
    denominator = (
        2.0
        * area.to(device=prediction.device, dtype=prediction.dtype)[:, None]
        * variances[None, :]
    ).clamp(min=eps)
    exponent = squared_distance / denominator
    similarity = torch.exp(-exponent) * visible.to(prediction.dtype)
    visible_count = visible.sum(dim=-1)
    valid_instances = visible_count > 0
    if not valid_instances.any():
        return prediction.sum() * 0
    oks = similarity.sum(dim=-1) / visible_count.clamp(min=1)
    return (1.0 - oks[valid_instances]).mean()


class PersonMaskDecoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        visual_hidden_size: int,
        decoder_dim: int = 256,
    ) -> None:
        super().__init__()
        groups = 8 if decoder_dim % 8 == 0 else 1
        self.seg_project = nn.Linear(hidden_size, decoder_dim)
        self.logit_scale = float(decoder_dim) ** -0.5
        self.parent_project = nn.Linear(hidden_size, decoder_dim)
        self.visual_project = nn.Linear(visual_hidden_size, decoder_dim)
        self.pixel_decoder = nn.Sequential(
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
            nn.GroupNorm(groups, decoder_dim),
            nn.GELU(),
            nn.Conv2d(decoder_dim, decoder_dim, 3, padding=1),
        )

    def forward(
        self,
        seg_states: torch.Tensor,
        parent_states: torch.Tensor,
        visual_tokens: torch.Tensor,
        *,
        spatial_shape: Tuple[int, int],
    ) -> torch.Tensor:
        persons = int(seg_states.shape[0])
        if parent_states.shape[0] != persons:
            raise ValueError("seg and parent states must have equal person counts")
        if visual_tokens.ndim == 3:
            if visual_tokens.shape[0] != 1:
                raise ValueError("mask decoder processes one image at a time")
            visual_tokens = visual_tokens[0]
        height, width = [int(value) for value in spatial_shape]
        if visual_tokens.shape[0] != height * width:
            raise ValueError("visual token count does not match spatial shape")
        pixels = self.visual_project(visual_tokens)
        pixels = pixels.transpose(0, 1).reshape(1, -1, height, width)
        pixels = F.interpolate(
            pixels, scale_factor=4.0, mode="bilinear", align_corners=False
        )
        pixels = self.pixel_decoder(pixels)[0]
        if persons == 0:
            return pixels.new_zeros((0, pixels.shape[-2], pixels.shape[-1]))
        masks = self.seg_project(seg_states) + self.parent_project(parent_states)
        return torch.einsum("nd,dhw->nhw", masks, pixels) * self.logit_scale


def mask_bce_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if logits.shape[0] != targets.shape[0]:
        raise ValueError("mask predictions and targets must have equal counts")
    if logits.shape[0] == 0:
        zero = logits.sum() * 0
        return zero, zero
    target = targets.to(device=logits.device, dtype=logits.dtype).unsqueeze(1)
    target = F.interpolate(target, size=logits.shape[-2:], mode="nearest")[:, 0]
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits)
    numerator = 2.0 * (probability * target).flatten(1).sum(dim=1) + 1.0
    denominator = (
        probability.flatten(1).sum(dim=1)
        + target.flatten(1).sum(dim=1)
        + 1.0
    )
    dice = (1.0 - numerator / denominator).mean()
    return bce, dice


def require_exact_token_states(
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    labels: Optional[torch.Tensor],
    token_id: int,
    expected_counts: Sequence[int],
    task_name: str,
) -> List[torch.Tensor]:
    if len(expected_counts) != input_ids.shape[0]:
        raise ValueError("expected count list must match batch size")
    result: List[torch.Tensor] = []
    for batch_index, expected in enumerate(expected_counts):
        positions = input_ids[batch_index].eq(int(token_id))
        if labels is not None:
            positions = positions & labels[batch_index].ne(-100)
        states = hidden[batch_index][positions]
        actual = int(states.shape[0])
        if actual != int(expected):
            raise RuntimeError(
                f"{task_name} token/target mismatch at batch={batch_index}: "
                f"tokens={actual} targets={int(expected)}"
            )
        result.append(states)
    return result

