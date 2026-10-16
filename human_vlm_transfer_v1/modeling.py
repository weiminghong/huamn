from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from human_lisa_qwen_v4.losses_v4 import task_normalized_causal_loss


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
        0.107,
        0.107,
        0.087,
        0.087,
        0.089,
        0.089,
        0.089,
    ],
    dtype=torch.float32,
)


def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center = boxes[..., :2]
    half_size = boxes[..., 2:].clamp(min=1e-4) * 0.5
    return torch.cat([center - half_size, center + half_size], dim=-1).clamp(0.0, 1.0)


def xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(
        [(boxes[..., :2] + boxes[..., 2:]) * 0.5, boxes[..., 2:] - boxes[..., :2]],
        dim=-1,
    )


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    size = (boxes[..., 2:] - boxes[..., :2]).clamp(min=0.0)
    return size[..., 0] * size[..., 1]


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod(dim=-1)
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - intersection
    iou = intersection / union.clamp(min=1e-6)

    enclosing_tl = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_br = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing = (enclosing_br - enclosing_tl).clamp(min=0.0).prod(dim=-1)
    return iou - (enclosing - union) / enclosing.clamp(min=1e-6)


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    probability = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    modulation = torch.where(targets > 0.5, 1.0 - probability, probability).pow(gamma)
    alpha_weight = torch.where(
        targets > 0.5,
        logits.new_tensor(alpha),
        logits.new_tensor(1.0 - alpha),
    )
    return (ce * modulation * alpha_weight).mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    probabilities = logits.sigmoid().flatten(1)
    targets = targets.flatten(1)
    numerator = 2.0 * (probabilities * targets).sum(dim=1)
    denominator = probabilities.sum(dim=1) + targets.sum(dim=1)
    return (1.0 - (numerator + 1.0) / (denominator + 1.0)).mean()


def unshuffle_premerger_tokens(
    tokens: torch.Tensor,
    *,
    grid_thw: Sequence[int],
    merge_size: int,
) -> torch.Tensor:
    temporal, grid_h, grid_w = [int(value) for value in grid_thw]
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError(f"grid {(grid_h, grid_w)} is not divisible by {merge_size}")
    expected = temporal * grid_h * grid_w
    if tokens.shape[0] != expected:
        raise ValueError(f"token/grid mismatch: {tokens.shape[0]} vs {expected}")
    channels = tokens.shape[-1]
    feature = tokens.reshape(
        temporal,
        grid_h // merge_size,
        grid_w // merge_size,
        merge_size,
        merge_size,
        channels,
    )
    feature = feature.permute(0, 1, 3, 2, 4, 5).reshape(
        temporal, grid_h, grid_w, channels
    )
    if temporal == 1:
        feature = feature[0]
    else:
        feature = feature.mean(dim=0)
    feature = feature.permute(2, 0, 1).contiguous()
    return feature


def split_premerger_maps(
    tokens: torch.Tensor,
    image_grid_thw: torch.Tensor,
    *,
    merge_size: int,
) -> List[torch.Tensor]:
    maps = []
    offset = 0
    for row in image_grid_thw.detach().cpu().tolist():
        count = int(row[0]) * int(row[1]) * int(row[2])
        maps.append(
            unshuffle_premerger_tokens(
                tokens[offset : offset + count],
                grid_thw=row,
                merge_size=merge_size,
            )
        )
        offset += count
    if offset != tokens.shape[0]:
        raise ValueError(f"unused vision tokens: consumed {offset} of {tokens.shape[0]}")
    return maps


def crop_feature_rois(
    feature: torch.Tensor,
    boxes: torch.Tensor,
    *,
    output_size: Tuple[int, int] = (32, 24),
) -> torch.Tensor:
    """Differentiably crop one normalized box per person from a single image."""
    if feature.shape[0] != 1:
        raise ValueError("ROI crop currently expects one image at a time")
    if boxes.shape[0] == 0:
        return feature.new_zeros((0, feature.shape[1], *output_size))
    out_h, out_w = [int(value) for value in output_size]
    y = (
        torch.arange(out_h, device=feature.device, dtype=feature.dtype) + 0.5
    ) / out_h
    x = (
        torch.arange(out_w, device=feature.device, dtype=feature.dtype) + 0.5
    ) / out_w
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    box_width = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-4)
    box_height = (boxes[:, 3] - boxes[:, 1]).clamp(min=1e-4)
    sample_x = boxes[:, 0, None, None] + grid_x[None] * box_width[:, None, None]
    sample_y = boxes[:, 1, None, None] + grid_y[None] * box_height[:, None, None]
    grid = torch.stack([sample_x * 2.0 - 1.0, sample_y * 2.0 - 1.0], dim=-1)
    return F.grid_sample(
        feature.expand(boxes.shape[0], -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, layers: int):
        super().__init__()
        modules: List[nn.Module] = []
        for index in range(layers):
            in_dim = input_dim if index == 0 else hidden_dim
            out_dim = output_dim if index == layers - 1 else hidden_dim
            modules.append(nn.Linear(in_dim, out_dim))
            if index != layers - 1:
                modules.append(nn.GELU())
        self.net = nn.Sequential(*modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CoordinateEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.project = MLP(2, dim, dim, 2)

    def forward(self, height: int, width: int, reference: torch.Tensor) -> torch.Tensor:
        y = (torch.arange(height, device=reference.device, dtype=reference.dtype) + 0.5) / height
        x = (torch.arange(width, device=reference.device, dtype=reference.dtype) + 0.5) / width
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
        return self.project(coordinates)


class SpatialAdapter(nn.Module):
    def __init__(self, in_channels: int, dim: int, num_inputs: int):
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_channels, dim, 1),
                    nn.GroupNorm(32, dim),
                    nn.GELU(),
                )
                for _ in range(num_inputs)
            ]
        )
        self.refine16 = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.GELU(),
        )
        self.up8 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 2, stride=2),
            nn.GroupNorm(32, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GELU(),
        )

    def forward(self, maps: Sequence[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
        if len(maps) != len(self.projections):
            raise ValueError("unexpected number of vision maps")
        projected = [
            projection(feature.unsqueeze(0))
            for projection, feature in zip(self.projections, maps)
        ]
        p16 = self.refine16(torch.stack(projected, dim=0).mean(dim=0))
        p8 = self.up8(p16)
        p4 = self.up4(p8)
        return p16, p8, p4


class HumanQueryDecoder(nn.Module):
    def __init__(self, dim: int, num_queries: int, num_layers: int, num_heads: int):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.position = CoordinateEmbedding(dim)
        self.layers = nn.ModuleList(
            [
                nn.TransformerDecoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, feature: torch.Tensor, human_context: torch.Tensor) -> torch.Tensor:
        _, channels, height, width = feature.shape
        memory = feature.flatten(2).transpose(1, 2)
        memory = memory + self.position(height, width, feature)[None]
        query = self.queries[None].to(feature) + human_context[:, None]
        for layer in self.layers:
            query = layer(query, memory)
        return self.norm(query)


class PoseCrossLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = MLP(dim, dim * 4, dim, 2)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, query: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        value, _ = self.self_attention(query, query, query, need_weights=False)
        query = self.norm1(query + value)
        value, _ = self.cross_attention(query, memory, memory, need_weights=False)
        query = self.norm2(query + value)
        return self.norm3(query + self.ffn(query))


class BoxConditionedPoseHead(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.joint_embedding = nn.Parameter(torch.randn(17, dim) * 0.02)
        self.position = CoordinateEmbedding(dim)
        self.layers = nn.ModuleList(
            [PoseCrossLayer(dim, num_heads) for _ in range(num_layers)]
        )
        self.coordinate = nn.Linear(dim, 2)
        self.visibility = nn.Linear(dim, 3)
        self.heatmap_query = nn.Linear(dim, dim)

    def forward(
        self,
        person_queries: torch.Tensor,
        feature: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if person_queries.shape[0] == 0:
            height, width = feature.shape[-2:]
            return (
                person_queries.new_zeros((0, 17, 2)),
                person_queries.new_zeros((0, 17, 3)),
                person_queries.new_zeros((0, 17, height, width)),
            )
        _, _, height, width = feature.shape
        if feature.shape[0] == 1:
            pose_feature = feature.expand(person_queries.shape[0], -1, -1, -1)
        elif feature.shape[0] == person_queries.shape[0]:
            pose_feature = feature
        else:
            raise ValueError(
                "pose feature batch must be one or match the person count"
            )
        memory = pose_feature.flatten(2).transpose(1, 2)
        memory = memory + self.position(height, width, feature)[None]
        query = person_queries[:, None] + self.joint_embedding[None].to(person_queries)
        for layer in self.layers:
            query = layer(query, memory)
        relative = torch.sigmoid(self.coordinate(query))
        visibility = self.visibility(query)
        heatmaps = torch.einsum(
            "mjd,mdhw->mjhw",
            self.heatmap_query(query),
            pose_feature,
        ) * (query.shape[-1] ** -0.5)
        return relative, visibility, heatmaps


@dataclass
class Match:
    prediction_indices: torch.Tensor
    target_indices: torch.Tensor


class Stage1TransferModel(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        vision_module: nn.Module,
        vision_layer_indices: Sequence[int],
        vision_hidden_size: int,
        text_hidden_size: int,
        merge_size: int,
        anchor_token_ids: Mapping[str, int],
        num_queries: int = 100,
        decoder_dim: int = 256,
        decoder_layers: int = 3,
        decoder_heads: int = 8,
        text_task_weights: Mapping[str, float] | None = None,
    ):
        super().__init__()
        self.base_model = base_model
        self.vision_module = vision_module
        self.vision_layer_indices = tuple(int(v) for v in vision_layer_indices)
        self.merge_size = int(merge_size)
        self.anchor_token_ids = {key: int(value) for key, value in anchor_token_ids.items()}
        self.text_task_weights = dict(text_task_weights or {})
        self._captured_vision: Dict[int, torch.Tensor] = {}
        self._hook_handles = []
        for index in self.vision_layer_indices:
            handle = self.vision_module.blocks[index].register_forward_hook(
                self._capture_hook(index)
            )
            self._hook_handles.append(handle)

        self.spatial_adapter = SpatialAdapter(
            vision_hidden_size, decoder_dim, len(self.vision_layer_indices)
        )
        self.task_projectors = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.LayerNorm(text_hidden_size),
                    nn.Linear(text_hidden_size, decoder_dim),
                )
                for name in self.anchor_token_ids
            }
        )
        self.query_decoder = HumanQueryDecoder(
            decoder_dim, num_queries, decoder_layers, decoder_heads
        )
        self.object_head = nn.Linear(decoder_dim, 1)
        self.box_head = MLP(decoder_dim, decoder_dim, 4, 3)
        self.gender_head = nn.Linear(decoder_dim, 2)
        self.age_head = nn.Linear(decoder_dim, 121)
        self.mask_embed = MLP(decoder_dim, decoder_dim, decoder_dim, 3)
        self.pose_head = BoxConditionedPoseHead(decoder_dim, decoder_heads, 2)

    def _capture_hook(self, index: int):
        def hook(_module, _args, output):
            if isinstance(output, (tuple, list)):
                output = output[0]
            self._captured_vision[index] = output
        return hook

    def _task_contexts(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        batch_index: int,
    ) -> Dict[str, torch.Tensor]:
        contexts = {}
        for name, token_id in self.anchor_token_ids.items():
            selected = hidden[batch_index][input_ids[batch_index].eq(token_id)]
            if selected.shape[0] != 1:
                raise RuntimeError(
                    f"anchor {name} must occur once, found {selected.shape[0]}"
                )
            contexts[name] = self.task_projectors[name](selected)
        return contexts

    def _match(
        self,
        object_logits: torch.Tensor,
        boxes: torch.Tensor,
        target_boxes: torch.Tensor,
    ) -> Match:
        if target_boxes.shape[0] == 0:
            empty = torch.empty(0, device=boxes.device, dtype=torch.long)
            return Match(empty, empty)
        if target_boxes.shape[0] > boxes.shape[0]:
            raise RuntimeError("ground-truth persons exceed human query count")
        with torch.no_grad():
            l1 = torch.cdist(
                xyxy_to_cxcywh(boxes.float()),
                xyxy_to_cxcywh(target_boxes.float()),
                p=1,
            )
            giou = generalized_box_iou(boxes.float(), target_boxes.float())
            confidence = object_logits.float().sigmoid()[:, None]
            cost = 5.0 * l1 - 2.0 * giou - confidence
            pred, target = linear_sum_assignment(cost.cpu().numpy())
        return Match(
            torch.as_tensor(pred, device=boxes.device, dtype=torch.long),
            torch.as_tensor(target, device=boxes.device, dtype=torch.long),
        )

    def _mask_losses(
        self,
        task_queries: torch.Tensor,
        p4: torch.Tensor,
        match: Match,
        target: Mapping[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        prediction_indices = []
        targets = []
        for pred_index, target_index in zip(
            match.prediction_indices.tolist(), match.target_indices.tolist()
        ):
            if not bool(target["seg_valid"][target_index]):
                continue
            mask = target["masks"][target_index]
            if mask is None:
                continue
            prediction_indices.append(pred_index)
            targets.append(mask)
        if not prediction_indices:
            zero = p4.sum() * 0.0
            return zero, zero, 0
        selected = task_queries[prediction_indices]
        logits = torch.einsum(
            "md,bdhw->mbhw", self.mask_embed(selected), p4
        )[:, 0] * (selected.shape[-1] ** -0.5)
        resized = torch.stack(
            [
                F.interpolate(
                    mask[None, None].to(device=logits.device, dtype=torch.float32),
                    size=logits.shape[-2:],
                    mode="nearest",
                )[0, 0]
                for mask in targets
            ],
            dim=0,
        ).to(logits.dtype)
        return (
            F.binary_cross_entropy_with_logits(logits, resized),
            dice_loss(logits, resized),
            len(prediction_indices),
        )

    def _pose_losses(
        self,
        task_queries: torch.Tensor,
        p8: torch.Tensor,
        predicted_boxes: torch.Tensor,
        match: Match,
        target: Mapping[str, Any],
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        pred_indices = []
        target_indices = []
        for pred_index, target_index in zip(
            match.prediction_indices.tolist(), match.target_indices.tolist()
        ):
            if bool(target["pose_valid"][target_index]):
                pred_indices.append(pred_index)
                target_indices.append(target_index)
        if not pred_indices:
            zero = p8.sum() * 0.0
            return {
                "pose_relative": zero,
                "pose_absolute": zero,
                "pose_visibility": zero,
                "pose_oks": zero,
                "pose_heatmap": zero,
            }, 0

        del predicted_boxes
        queries = task_queries[pred_indices]
        target_pose = target["poses"][target_indices].to(
            device=queries.device, dtype=queries.dtype
        )
        target_boxes = target["boxes"][target_indices].to(
            device=queries.device, dtype=queries.dtype
        )
        roi_feature = crop_feature_rois(
            p8,
            target_boxes,
            output_size=(32, 24),
        )
        relative, visibility_logits, heatmaps = self.pose_head(
            queries, roi_feature
        )
        visible = target_pose[..., 2] > 0
        box_size = (target_boxes[:, None, 2:] - target_boxes[:, None, :2]).clamp(
            min=1e-4
        )
        target_relative = (
            target_pose[..., :2] - target_boxes[:, None, :2]
        ) / box_size
        target_relative = target_relative.clamp(0.0, 1.0)
        relative_l1 = F.l1_loss(relative[visible], target_relative[visible])

        absolute = target_boxes[:, None, :2] + relative * box_size
        absolute_l1 = F.l1_loss(absolute[visible], target_pose[..., :2][visible])
        visibility_loss = F.cross_entropy(
            visibility_logits.reshape(-1, 3),
            target_pose[..., 2].long().reshape(-1),
        )

        image_h, image_w = target["image_size"]
        pixel_delta = (absolute.float() - target_pose[..., :2].float()).square()
        pixel_delta[..., 0] *= float(image_w * image_w)
        pixel_delta[..., 1] *= float(image_h * image_h)
        squared_distance = pixel_delta.sum(dim=-1)
        area = target["areas"][target_indices].to(
            device=queries.device, dtype=torch.float32
        )[:, None].clamp(min=1.0)
        sigmas = COCO_KEYPOINT_SIGMAS.to(device=queries.device)[None]
        exponent = squared_distance / (2.0 * area * (2.0 * sigmas).square())
        oks = torch.exp(-exponent)
        oks_loss = 1.0 - oks[visible].mean()

        heat_h, heat_w = heatmaps.shape[-2:]
        target_x = (target_relative[..., 0] * heat_w).long().clamp(0, heat_w - 1)
        target_y = (target_relative[..., 1] * heat_h).long().clamp(0, heat_h - 1)
        heat_target = target_y * heat_w + target_x
        raw_heatmap_loss = F.cross_entropy(
            heatmaps.flatten(2)[visible],
            heat_target[visible],
        )
        heatmap_loss = raw_heatmap_loss / torch.log(
            heatmaps.new_tensor(float(heat_h * heat_w))
        )
        return {
            "pose_relative": relative_l1,
            "pose_absolute": absolute_l1,
            "pose_visibility": visibility_loss,
            "pose_oks": oks_loss,
            "pose_heatmap": heatmap_loss,
        }, len(pred_indices)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        text_task_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        dense_targets: Sequence[Mapping[str, Any]],
        **base_kwargs: Any,
    ) -> Dict[str, Any]:
        self._captured_vision.clear()
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **base_kwargs,
        )
        text_loss, text_components, text_counts = task_normalized_causal_loss(
            outputs.logits,
            labels,
            text_task_ids,
            task_weights=self.text_task_weights,
        )
        hidden = outputs.hidden_states[-1]
        missing = [
            index for index in self.vision_layer_indices
            if index not in self._captured_vision
        ]
        if missing:
            raise RuntimeError(f"vision hooks did not fire: {missing}")
        maps_by_layer = {
            index: split_premerger_maps(
                self._captured_vision[index],
                image_grid_thw,
                merge_size=self.merge_size,
            )
            for index in self.vision_layer_indices
        }

        loss_lists: Dict[str, List[torch.Tensor]] = {
            "object": [],
            "box_l1": [],
            "box_giou": [],
            "mask_bce": [],
            "mask_dice": [],
            "pose_relative": [],
            "pose_absolute": [],
            "pose_visibility": [],
            "pose_oks": [],
            "pose_heatmap": [],
            "gender": [],
            "age": [],
        }
        matches = 0
        mask_labels = 0
        pose_labels = 0
        feature_shapes = []

        for batch_index, target_source in enumerate(dense_targets):
            target = dict(target_source)
            for key in (
                "boxes",
                "poses",
                "pose_valid",
                "seg_valid",
                "genders",
                "gender_valid",
                "ages",
                "age_valid",
                "areas",
            ):
                target[key] = target[key].to(hidden.device)
            layer_maps = [
                maps_by_layer[index][batch_index]
                for index in self.vision_layer_indices
            ]
            p16, p8, p4 = self.spatial_adapter(layer_maps)
            feature_shapes.append(
                {
                    "p16": list(p16.shape),
                    "p8": list(p8.shape),
                    "p4": list(p4.shape),
                }
            )
            contexts = self._task_contexts(input_ids, hidden, batch_index)
            queries = self.query_decoder(p16, contexts["human"])
            object_query = queries[0] + contexts["box"]
            object_logits = self.object_head(object_query).squeeze(-1)
            box_cxcywh = torch.sigmoid(self.box_head(object_query))
            boxes = cxcywh_to_xyxy(box_cxcywh)
            match = self._match(object_logits, boxes, target["boxes"])
            matches += int(match.prediction_indices.numel())

            object_targets = torch.zeros_like(object_logits)
            object_targets[match.prediction_indices] = 1.0
            loss_lists["object"].append(
                sigmoid_focal_loss(object_logits, object_targets)
            )
            if match.prediction_indices.numel():
                pred_boxes = boxes[match.prediction_indices]
                gt_boxes = target["boxes"][match.target_indices]
                loss_lists["box_l1"].append(
                    F.l1_loss(
                        xyxy_to_cxcywh(pred_boxes),
                        xyxy_to_cxcywh(gt_boxes),
                    )
                )
                giou = generalized_box_iou(pred_boxes, gt_boxes).diag()
                loss_lists["box_giou"].append((1.0 - giou).mean())

                mask_queries = queries[0] + contexts["mask"]
                mask_bce, mask_dice_value, mask_count = self._mask_losses(
                    mask_queries, p4, match, target
                )
                loss_lists["mask_bce"].append(mask_bce)
                loss_lists["mask_dice"].append(mask_dice_value)
                mask_labels += mask_count

                pose_queries = queries[0] + contexts["pose"]
                pose_values, pose_count = self._pose_losses(
                    pose_queries, p8, boxes, match, target
                )
                for key, value in pose_values.items():
                    loss_lists[key].append(value)
                pose_labels += pose_count

                pred_idx = match.prediction_indices
                tgt_idx = match.target_indices
                valid_gender = target["gender_valid"][tgt_idx]
                if valid_gender.any():
                    gender_logits = self.gender_head(
                        queries[0, pred_idx] + contexts["gender"]
                    )
                    loss_lists["gender"].append(
                        F.cross_entropy(
                            gender_logits[valid_gender],
                            target["genders"][tgt_idx][valid_gender],
                        )
                    )
                valid_age = target["age_valid"][tgt_idx]
                if valid_age.any():
                    age_logits = self.age_head(
                        queries[0, pred_idx] + contexts["age"]
                    )
                    loss_lists["age"].append(
                        F.cross_entropy(
                            age_logits[valid_age],
                            target["ages"][tgt_idx][valid_age],
                        )
                    )

        reference = text_loss
        reduced = {
            name: (
                torch.stack(values).mean()
                if values
                else reference * 0.0
            )
            for name, values in loss_lists.items()
        }
        dense_loss = (
            2.0 * reduced["object"]
            + 5.0 * reduced["box_l1"]
            + 2.0 * reduced["box_giou"]
            + 2.0 * reduced["mask_bce"]
            + 2.0 * reduced["mask_dice"]
            + 5.0 * reduced["pose_relative"]
            + 2.0 * reduced["pose_absolute"]
            + 0.5 * reduced["pose_visibility"]
            + 2.0 * reduced["pose_oks"]
            + 5.0 * reduced["pose_heatmap"]
            + 1.0 * reduced["gender"]
            + 0.25 * reduced["age"]
        )
        total = text_loss + dense_loss
        result: Dict[str, Any] = {
            "loss": total,
            "text_loss": text_loss,
            "dense_loss": dense_loss,
            "matched_persons": matches,
            "mask_labels": mask_labels,
            "pose_labels": pose_labels,
            "feature_shapes": feature_shapes,
            "text_counts": text_counts,
        }
        result.update({f"loss_{key}": value for key, value in reduced.items()})
        result.update(
            {f"text_{key}": value for key, value in text_components.items()}
        )
        return result
