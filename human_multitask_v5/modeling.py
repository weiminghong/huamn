from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Mask2FormerConfig
from human_lisa_pose_stage1.modeling import split_premerger_maps

from .losses import parsing_loss, segment_loss

from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerPixelDecoder,
    Mask2FormerTransformerModule,
)


@dataclass
class DenseOutputs:
    segment_logits: torch.Tensor
    parsing_logits: torch.Tensor
    feature_shapes: Dict[str, List[int]]


class QwenFeaturePyramid(nn.Module):
    """Turn equal-resolution intermediate ViT maps into P4/P8/P16/P32."""

    def __init__(
        self,
        input_channels: Sequence[int],
        hidden_dim: int = 256,
    ) -> None:
        super().__init__()
        if len(input_channels) < 2:
            raise ValueError("at least two Qwen layers are required")
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(int(channels), hidden_dim, 1, bias=False),
                    nn.GroupNorm(32, hidden_dim),
                    nn.GELU(),
                )
                for channels in input_channels
            ]
        )
        self.layer_weights = nn.Parameter(
            torch.zeros(len(input_channels), dtype=torch.float32)
        )
        self.refine = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False),
            nn.GroupNorm(32, hidden_dim),
            nn.GELU(),
        )
        self.p8_refine = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.p4_refine = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        self.p32_refine = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)

    def forward(self, layer_maps: Sequence[torch.Tensor]) -> List[torch.Tensor]:
        if len(layer_maps) != len(self.projections):
            raise ValueError("layer map count does not match configured projections")
        shape = layer_maps[0].shape[-2:]
        if any(value.shape[-2:] != shape for value in layer_maps):
            raise ValueError("Qwen layer maps must be in one representation grid")
        weights = self.layer_weights.softmax(0)
        fused = sum(
            weight * project(feature)
            for weight, project, feature in zip(
                weights, self.projections, layer_maps
            )
        )
        p16 = self.refine(fused)
        p8 = self.p8_refine(
            F.interpolate(p16, scale_factor=2.0, mode="bilinear", align_corners=False)
        )
        p4 = self.p4_refine(
            F.interpolate(p8, scale_factor=2.0, mode="bilinear", align_corners=False)
        )
        p32 = self.p32_refine(F.avg_pool2d(p16, kernel_size=2, stride=2))
        return [p4, p8, p16, p32]


class PersonConditionedMask2Former(nn.Module):
    """Official Mask2Former modules with person-owned LLM query content.

    This is an interface adaptation, not a reproduction of a published complete
    model. No RGB image encoder is present in this class.
    """

    def __init__(
        self,
        *,
        text_dim: int,
        qwen_feature_channels: Sequence[int],
        hidden_dim: int = 256,
        num_parts: int = 58,
        encoder_layers: int = 2,
        decoder_layers: int = 4,
        attention_heads: int = 8,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.num_parts = int(num_parts)
        self.pyramid = QwenFeaturePyramid(
            qwen_feature_channels, hidden_dim=self.hidden_dim
        )
        config = Mask2FormerConfig(
            feature_size=self.hidden_dim,
            mask_feature_size=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            encoder_layers=int(encoder_layers),
            decoder_layers=int(decoder_layers) + 1,
            num_attention_heads=int(attention_heads),
            feature_strides=[4, 8, 16, 32],
            use_auxiliary_loss=True,
        )
        self.config = config
        self.pixel_decoder = Mask2FormerPixelDecoder(
            config, [self.hidden_dim] * 4
        )
        self.transformer = Mask2FormerTransformerModule(
            self.hidden_dim, config
        )
        self.owner_project = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.box_project = nn.Sequential(
            nn.Linear(4, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.segment_embedding = nn.Parameter(
            torch.randn(1, self.hidden_dim) * 0.02
        )
        self.part_embeddings = nn.Parameter(
            torch.randn(self.num_parts, self.hidden_dim) * 0.02
        )
        self.query_position = nn.Sequential(
            nn.LayerNorm(self.hidden_dim),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    def _decode_queries(
        self,
        multi_scale_features: Sequence[torch.Tensor],
        mask_features: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        if queries.ndim != 2:
            raise ValueError("queries must be [num_queries, hidden_dim]")
        if queries.shape[0] == 0:
            return mask_features.new_zeros(
                (0, *mask_features.shape[-2:])
            )
        module = self.transformer
        stage_features = []
        stage_positions = []
        size_list = []
        for index in range(module.num_feature_levels):
            feature = multi_scale_features[index]
            size_list.append(feature.shape[-2:])
            position = module.position_embedder(
                feature.shape, feature.device, feature.dtype, None
            ).flatten(2)
            encoded = (
                module.input_projections[index](feature).flatten(2)
                + module.level_embed.weight[index][None, :, None]
            )
            stage_positions.append(position.permute(2, 0, 1))
            stage_features.append(encoded.permute(2, 0, 1))
        content = queries[:, None, :]
        positions = self.query_position(queries)[:, None, :]
        output = module.decoder(
            inputs_embeds=content,
            multi_stage_positional_embeddings=stage_positions,
            pixel_embeddings=mask_features,
            encoder_hidden_states=stage_features,
            query_position_embeddings=positions,
            feature_size_list=size_list,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )
        return output.masks_queries_logits[-1][0]

    def forward(
        self,
        *,
        layer_maps: Sequence[torch.Tensor],
        owner_states: torch.Tensor,
        boxes_normalized: torch.Tensor,
        segment_owner_indices: torch.Tensor,
        parsing_owner_indices: torch.Tensor,
    ) -> DenseOutputs:
        if owner_states.ndim != 2 or boxes_normalized.shape != (
            owner_states.shape[0], 4
        ):
            raise ValueError("owner states and boxes have incompatible shapes")
        pyramid = self.pyramid(layer_maps)
        pixel = self.pixel_decoder(pyramid, return_dict=True)
        owners = self.owner_project(owner_states) + self.box_project(
            boxes_normalized.float()
        )

        seg_owners = owners.index_select(0, segment_owner_indices)
        seg_queries = seg_owners + self.segment_embedding
        segment_logits = self._decode_queries(
            pixel.multi_scale_features, pixel.mask_features, seg_queries
        )

        parse_owners = owners.index_select(0, parsing_owner_indices)
        parsing_queries = (
            parse_owners[:, None, :] + self.part_embeddings[None, :, :]
        ).reshape(-1, self.hidden_dim)
        parsing_flat = self._decode_queries(
            pixel.multi_scale_features, pixel.mask_features, parsing_queries
        )
        parsing_logits = parsing_flat.reshape(
            parse_owners.shape[0],
            self.num_parts,
            *parsing_flat.shape[-2:],
        )
        return DenseOutputs(
            segment_logits=segment_logits,
            parsing_logits=parsing_logits,
            feature_shapes={
                "p4": list(pyramid[0].shape),
                "p8": list(pyramid[1].shape),
                "p16": list(pyramid[2].shape),
                "p32": list(pyramid[3].shape),
                "mask": list(pixel.mask_features.shape),
            },
        )

    def load_shape_compatible_official_weights(
        self, state_dict: Mapping[str, torch.Tensor]
    ) -> Dict[str, int]:
        """Load only exact-name, exact-shape official decoder parameters."""
        own = self.state_dict()
        compatible = {
            key: value
            for key, value in state_dict.items()
            if key in own and own[key].shape == value.shape
        }
        result = self.load_state_dict(compatible, strict=False)
        return {
            "loaded": len(compatible),
            "missing": len(result.missing_keys),
            "unexpected": len(result.unexpected_keys),
        }


def require_owner_states(
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    labels: torch.Tensor,
    human_token_id: int,
    expected_counts: Sequence[int],
) -> List[torch.Tensor]:
    rows = []
    for row, expected in enumerate(expected_counts):
        mask = input_ids[row].eq(int(human_token_id)) & labels[row].ne(-100)
        selected = hidden[row][mask]
        if int(selected.shape[0]) != int(expected):
            raise RuntimeError(
                f"human owner mismatch row={row} "
                f"states={selected.shape[0]} expected={expected}"
            )
        rows.append(selected)
    return rows


class Stage1V5Model(nn.Module):
    """Qwen text tasks plus person-conditioned Mask2Former dense tasks."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        vision_module: nn.Module,
        vision_layer_indices: Sequence[int],
        vision_hidden_size: int,
        text_hidden_size: int,
        merge_size: int,
        human_token_id: int,
        decoder_dim: int = 256,
        decoder_encoder_layers: int = 2,
        decoder_layers: int = 4,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.vision_module = vision_module
        self.vision_layer_indices = tuple(
            int(index) for index in vision_layer_indices
        )
        self.merge_size = int(merge_size)
        self.human_token_id = int(human_token_id)
        self._captured_vision: Dict[int, torch.Tensor] = {}
        self._hook_handles = [
            self.vision_module.blocks[index].register_forward_hook(
                self._capture_hook(index)
            )
            for index in self.vision_layer_indices
        ]
        self.dense_decoder = PersonConditionedMask2Former(
            text_dim=text_hidden_size,
            qwen_feature_channels=[
                int(vision_hidden_size) for _ in self.vision_layer_indices
            ],
            hidden_dim=decoder_dim,
            num_parts=58,
            encoder_layers=decoder_encoder_layers,
            decoder_layers=decoder_layers,
            attention_heads=decoder_heads,
        )

    def _capture_hook(self, index: int):
        def hook(_module, _inputs, output):
            if isinstance(output, (list, tuple)):
                output = output[0]
            self._captured_vision[index] = output
        return hook

    @staticmethod
    def _reduce(
        values: List[torch.Tensor], reference: torch.Tensor
    ) -> torch.Tensor:
        if values:
            return torch.stack(values).mean()
        return reference.sum() * 0.0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
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
        shift_logits = outputs.logits[:, :-1].contiguous().float()
        shift_labels = labels[:, 1:].contiguous()
        text_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.shape[-1]),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        hidden = outputs.hidden_states[-1]
        person_counts = [
            int(target["boxes"].shape[0]) for target in dense_targets
        ]
        owners = require_owner_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            human_token_id=self.human_token_id,
            expected_counts=person_counts,
        )
        missing = [
            index
            for index in self.vision_layer_indices
            if index not in self._captured_vision
        ]
        if missing:
            raise RuntimeError(f"Qwen vision hooks did not fire: {missing}")
        maps_by_layer = {
            index: split_premerger_maps(
                self._captured_vision[index],
                image_grid_thw,
                merge_size=self.merge_size,
            )
            for index in self.vision_layer_indices
        }

        segment_bce_values: List[torch.Tensor] = []
        segment_dice_values: List[torch.Tensor] = []
        parsing_ce_values: List[torch.Tensor] = []
        parsing_dice_values: List[torch.Tensor] = []
        feature_shapes: List[Dict[str, List[int]]] = []
        segment_labels = 0
        parsing_labels = 0

        for row, raw_target in enumerate(dense_targets):
            segment_indices = raw_target["segment_owner_indices"].to(
                hidden.device
            )
            parsing_indices = raw_target["parsing_owner_indices"].to(
                hidden.device
            )
            if not segment_indices.numel() and not parsing_indices.numel():
                continue
            boxes = raw_target["boxes"].to(hidden.device)
            layer_maps = [
                maps_by_layer[index][row]
                for index in self.vision_layer_indices
            ]
            dense = self.dense_decoder(
                layer_maps=layer_maps,
                owner_states=owners[row],
                boxes_normalized=boxes,
                segment_owner_indices=segment_indices,
                parsing_owner_indices=parsing_indices,
            )
            feature_shapes.append(dense.feature_shapes)

            if segment_indices.numel():
                targets = torch.stack(raw_target["segment_masks"]).to(
                    hidden.device
                )
                targets = F.interpolate(
                    targets[:, None].float(),
                    size=dense.segment_logits.shape[-2:],
                    mode="nearest",
                )[:, 0]
                value = segment_loss(dense.segment_logits, targets)
                segment_bce_values.append(value["bce"])
                segment_dice_values.append(value["dice"])
                segment_labels += int(segment_indices.numel())

            if parsing_indices.numel():
                targets = torch.stack(raw_target["parsing_maps"]).to(
                    hidden.device
                )
                targets = F.interpolate(
                    targets[:, None].float(),
                    size=dense.parsing_logits.shape[-2:],
                    mode="nearest",
                )[:, 0].long()
                value = parsing_loss(dense.parsing_logits, targets)
                parsing_ce_values.append(value["ce"])
                parsing_dice_values.append(value["dice"])
                parsing_labels += int(parsing_indices.numel())

        segment_bce = self._reduce(segment_bce_values, text_loss)
        segment_dice = self._reduce(segment_dice_values, text_loss)
        parsing_ce = self._reduce(parsing_ce_values, text_loss)
        parsing_dice = self._reduce(parsing_dice_values, text_loss)
        dense_loss = (
            segment_bce + segment_dice + parsing_ce + parsing_dice
        )
        return {
            "loss": text_loss + dense_loss,
            "text_loss": text_loss,
            "dense_loss": dense_loss,
            "loss_segment_bce": segment_bce,
            "loss_segment_dice": segment_dice,
            "loss_parsing_ce": parsing_ce,
            "loss_parsing_dice": parsing_dice,
            "segment_labels": segment_labels,
            "parsing_labels": parsing_labels,
            "matched_persons": sum(person_counts),
            "feature_shapes": feature_shapes,
        }
