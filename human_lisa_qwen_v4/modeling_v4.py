from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .heads_v4 import (
    COCO_KEYPOINT_SIGMAS,
    BoxFreePoseDecoder,
    PersonMaskDecoderV4,
)
from .losses_v4 import (
    active_weighted_mean,
    mask_losses,
    pose_losses,
    task_normalized_causal_loss,
)
from .token_rows_v4 import SelectiveTokenRows


DEFAULT_COMPONENT_WEIGHTS = {
    "pose_l1": 1.0,
    "pose_oks": 1.0,
    "pose_visibility": 0.25,
    "mask_bce": 1.0,
    "mask_dice": 1.0,
}
DEFAULT_TASK_WEIGHTS = {"text": 1.0, "pose": 1.0, "seg": 1.0}


def _weighted_mean(
    values: List[Tuple[torch.Tensor, int]], reference: torch.Tensor
) -> torch.Tensor:
    denominator = sum(weight for _, weight in values)
    if denominator <= 0:
        return reference.sum() * 0.0
    return sum(value * weight for value, weight in values) / denominator


def require_record_token_states(
    *,
    input_ids: torch.Tensor,
    hidden: torch.Tensor,
    labels: Optional[torch.Tensor],
    token_id: int,
    expected_counts: Sequence[int],
    task_name: str,
    start_positions: Optional[Sequence[int]] = None,
) -> List[torch.Tensor]:
    if len(expected_counts) != input_ids.shape[0]:
        raise ValueError("expected counts must match batch size")
    states = []
    for row, expected in enumerate(expected_counts):
        mask = input_ids[row].eq(int(token_id))
        if labels is not None:
            mask = mask & labels[row].ne(-100)
        if start_positions is not None:
            position_mask = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).ge(int(start_positions[row]))
            mask = mask & position_mask
        selected = hidden[row][mask]
        if int(selected.shape[0]) != int(expected):
            raise RuntimeError(
                f"{task_name} record-token mismatch at batch row {row}: "
                f"states={selected.shape[0]} expected={expected}"
            )
        states.append(selected)
    return states


class Stage1V4Model(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        hidden_size: int,
        token_ids: Mapping[str, int],
        image_token_id: int,
        visual_merge_size: int = 2,
        decoder_dim: int = 256,
        decoder_heads: int = 8,
        decoder_layers: int = 2,
        text_task_weights: Optional[Mapping[str, float]] = None,
        component_weights: Optional[Mapping[str, float]] = None,
        task_weights: Optional[Mapping[str, float]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.token_ids = {name: int(value) for name, value in token_ids.items()}
        if set(self.token_ids) != {"pose", "seg"}:
            raise ValueError("v4 requires exactly pose and seg routing tokens")
        self.image_token_id = int(image_token_id)
        self.visual_merge_size = int(visual_merge_size)
        self.decoder_dim = int(decoder_dim)
        self.decoder_heads = int(decoder_heads)
        self.decoder_layers = int(decoder_layers)
        self.text_task_weights = dict(text_task_weights or {})
        self.component_weights = dict(DEFAULT_COMPONENT_WEIGHTS)
        if component_weights:
            self.component_weights.update(
                {name: float(value) for name, value in component_weights.items()}
            )
        self.task_weights = dict(DEFAULT_TASK_WEIGHTS)
        if task_weights:
            self.task_weights.update(
                {name: float(value) for name, value in task_weights.items()}
            )
        self.token_rows = SelectiveTokenRows(
            [self.token_ids["pose"], self.token_ids["seg"]], hidden_size
        )
        self.token_rows.attach(self.base_model)
        self.pose_decoder = BoxFreePoseDecoder(
            hidden_size,
            decoder_dim=decoder_dim,
            num_heads=decoder_heads,
            num_layers=decoder_layers,
        )
        self.mask_decoder = PersonMaskDecoderV4(
            hidden_size, decoder_dim=decoder_dim
        )
        if hasattr(self.base_model, "config"):
            self.base_model.config.use_cache = False

    def _visual_tokens_and_shape(
        self,
        input_ids: torch.Tensor,
        hidden: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_index: int,
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        visual = hidden[batch_index][
            input_ids[batch_index].eq(self.image_token_id)
        ]
        temporal, grid_height, grid_width = [
            int(value)
            for value in image_grid_thw[batch_index].detach().cpu().tolist()
        ]
        if temporal != 1:
            raise RuntimeError("v4 image training requires temporal grid 1")
        if (
            grid_height % self.visual_merge_size
            or grid_width % self.visual_merge_size
        ):
            raise RuntimeError("visual grid is not divisible by merge size")
        height = grid_height // self.visual_merge_size
        width = grid_width // self.visual_merge_size
        if int(visual.shape[0]) != height * width:
            raise RuntimeError(
                "visual token/grid mismatch: "
                f"tokens={visual.shape[0]} expected={height * width}"
            )
        return visual, (height, width)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        text_task_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        person_counts: Sequence[int],
        pose_targets: Sequence[torch.Tensor],
        pose_areas: Sequence[torch.Tensor],
        seg_targets: Sequence[torch.Tensor],
        pose_person_indices: Sequence[torch.Tensor],
        seg_person_indices: Sequence[torch.Tensor],
        original_sizes: Sequence[Tuple[int, int]],
        return_predictions: bool = False,
        **base_kwargs: Any,
    ) -> Dict[str, Any]:
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=None,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
            **base_kwargs,
        )
        hidden = outputs.hidden_states[-1]
        text_loss, text_components, text_counts = task_normalized_causal_loss(
            outputs.logits,
            labels,
            text_task_ids,
            task_weights=self.text_task_weights,
        )
        pose_records = require_record_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=self.token_ids["pose"],
            expected_counts=person_counts,
            task_name="pose",
        )
        seg_records = require_record_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=self.token_ids["seg"],
            expected_counts=person_counts,
            task_name="seg",
        )
        batch_size = int(input_ids.shape[0])
        target_groups = [
            pose_targets,
            pose_areas,
            seg_targets,
            pose_person_indices,
            seg_person_indices,
            original_sizes,
        ]
        if any(len(values) != batch_size for values in target_groups):
            raise ValueError("dense target groups must match batch size")

        pose_values: Dict[str, List[Tuple[torch.Tensor, int]]] = {
            "pose_l1": [],
            "pose_oks": [],
            "pose_visibility": [],
        }
        mask_values: Dict[str, List[Tuple[torch.Tensor, int]]] = {
            "mask_bce": [],
            "mask_dice": [],
        }
        predictions: List[Dict[str, torch.Tensor]] = []
        pose_count = 0
        seg_count = 0
        for batch_index in range(batch_size):
            visual_tokens, spatial_shape = self._visual_tokens_and_shape(
                input_ids, hidden, image_grid_thw, batch_index
            )
            pose_indices = pose_person_indices[batch_index].to(
                device=hidden.device, dtype=torch.long
            )
            selected_pose_states = pose_records[batch_index].index_select(
                0, pose_indices
            )
            pose_target = pose_targets[batch_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            areas = pose_areas[batch_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            coordinates, visibility_logits = self.pose_decoder(
                selected_pose_states,
                visual_tokens,
                spatial_shape=spatial_shape,
            )
            height, width = original_sizes[batch_index]
            image_sizes = hidden.new_tensor(
                [[height, width]], dtype=hidden.dtype
            ).expand(len(pose_target), -1)
            current_pose_losses = pose_losses(
                coordinates,
                visibility_logits,
                pose_target,
                areas,
                image_sizes,
                sigmas=COCO_KEYPOINT_SIGMAS,
            )
            current_pose_count = int(pose_target.shape[0])
            pose_count += current_pose_count
            for name, value in current_pose_losses.items():
                if current_pose_count:
                    pose_values[name].append((value, current_pose_count))

            seg_indices = seg_person_indices[batch_index].to(
                device=hidden.device, dtype=torch.long
            )
            selected_seg_states = seg_records[batch_index].index_select(
                0, seg_indices
            )
            mask_logits = self.mask_decoder(
                selected_seg_states,
                visual_tokens,
                spatial_shape=spatial_shape,
            )
            mask_target = seg_targets[batch_index].to(device=hidden.device)
            current_mask_losses = mask_losses(mask_logits, mask_target)
            current_seg_count = int(mask_target.shape[0])
            seg_count += current_seg_count
            for name, value in current_mask_losses.items():
                if current_seg_count:
                    mask_values[name].append((value, current_seg_count))

            if return_predictions:
                predictions.append(
                    {
                        "pose_coordinates": coordinates,
                        "pose_visibility_logits": visibility_logits,
                        "mask_logits": mask_logits,
                        "pose_person_indices": pose_indices,
                        "seg_person_indices": seg_indices,
                    }
                )

        pose_components = {
            name: _weighted_mean(values, hidden)
            for name, values in pose_values.items()
        }
        mask_components = {
            name: _weighted_mean(values, hidden)
            for name, values in mask_values.items()
        }
        pose_loss = active_weighted_mean(
            pose_components,
            self.component_weights,
            {name: pose_count > 0 for name in pose_components},
        )
        seg_loss = active_weighted_mean(
            mask_components,
            self.component_weights,
            {name: seg_count > 0 for name in mask_components},
        )
        task_losses = {"text": text_loss, "pose": pose_loss, "seg": seg_loss}
        total = active_weighted_mean(
            task_losses,
            self.task_weights,
            {"text": True, "pose": pose_count > 0, "seg": seg_count > 0},
        )
        result: Dict[str, Any] = {
            "loss": total,
            "text_loss": text_loss,
            "pose_loss": pose_loss,
            "seg_loss": seg_loss,
            "pose_person_count": pose_count,
            "seg_person_count": seg_count,
            "text_token_counts": text_counts,
        }
        result.update(
            {f"text_{name}_loss": value for name, value in text_components.items()}
        )
        result.update(pose_components)
        result.update(mask_components)
        if return_predictions:
            result["predictions"] = predictions
        return result

    def save_task_state(self, output_dir: str) -> None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "token_ids": self.token_ids,
                "image_token_id": self.image_token_id,
                "visual_merge_size": self.visual_merge_size,
                "decoder_dim": self.decoder_dim,
                "decoder_heads": self.decoder_heads,
                "decoder_layers": self.decoder_layers,
                "token_rows": self.token_rows.state_dict(),
                "pose_decoder": self.pose_decoder.state_dict(),
                "mask_decoder": self.mask_decoder.state_dict(),
                "text_task_weights": self.text_task_weights,
                "component_weights": self.component_weights,
                "task_weights": self.task_weights,
            },
            output / "stage1_v4_tasks.pt",
        )

    def load_task_state(self, checkpoint: str) -> None:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state["token_ids"] != self.token_ids:
            raise RuntimeError("v4 special token ids do not match checkpoint")
        self.token_rows.load_state_dict(state["token_rows"], strict=True)
        self.pose_decoder.load_state_dict(state["pose_decoder"], strict=True)
        self.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        self.text_task_weights = dict(state["text_task_weights"])
        self.component_weights = dict(state["component_weights"])
        self.task_weights = dict(state["task_weights"])
