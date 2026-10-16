from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .heads_v1 import (
    BoxQualityHead,
    DirectJointQueryPoseDecoder,
    MultiPersonPoseDecoder,
    TopDownHeatmapPoseDecoder,
    PersonMaskDecoder,
    aligned_box_iou,
    generalized_box_iou_loss,
    jitter_pose_boxes,
    keypoints_inside_boxes,
    keypoint_heatmap_mse_loss,
    mask_bce_dice_loss,
    oks_loss,
    require_exact_token_states,
    roi_pool_visual_tokens,
    roi_relative_pose_l1,
)


DEFAULT_LOSS_WEIGHTS = {
    "ce": 1.0,
    "box_l1": 5.0,
    "box_giou": 2.0,
    "box_quality": 1.0,
    "pose_l1": 5.0,
    "pose_oks": 1.5,
    "pose_heatmap": 0.0,
    "pose_visibility": 1.0,
    "mask_bce": 8.0,
    "mask_dice": 5.0,
}


def scale_task_gradient(value: torch.Tensor, scale: float) -> torch.Tensor:
    scale = float(scale)
    if not 0.0 <= scale <= 1.0:
        raise ValueError("task backbone gradient scale must be in [0, 1]")
    detached = value.detach()
    return detached + (value - detached) * scale

def select_pose_reference_boxes(
    predicted_boxes: torch.Tensor,
    target_boxes: torch.Tensor,
    *,
    training: bool,
) -> torch.Tensor:
    """Use the same detached predicted person reference at train and inference.

    Targets are retained in the signature for call-site compatibility and shape
    validation. They are never used to condition pose decoding: pose sees full
    image features and a predicted-person coordinate reference in both modes.
    """
    if predicted_boxes.shape != target_boxes.shape:
        raise ValueError("predicted and target pose boxes must have matching shapes")
    if predicted_boxes.ndim != 2 or predicted_boxes.shape[-1] != 4:
        raise ValueError("pose reference boxes must have shape Nx4")
    del training
    return predicted_boxes.detach()



def weighted_causal_language_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("language logits/labels must have shapes BxLxV and BxL")
    if weights is not None and weights.shape != labels.shape:
        raise ValueError("language loss weights must match labels")
    shift_logits = logits[:, :-1].float().contiguous()
    shift_labels = labels[:, 1:].contiguous()
    valid = shift_labels.ne(-100)
    token_loss = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.shape[-1]),
        shift_labels.reshape(-1),
        ignore_index=-100,
        reduction="none",
    ).reshape_as(shift_labels)
    if weights is None:
        effective_weight = valid.to(token_loss.dtype)
    else:
        effective_weight = weights[:, 1:].to(
            device=token_loss.device, dtype=token_loss.dtype
        )
        effective_weight = effective_weight * valid.to(token_loss.dtype)
    denominator = effective_weight.sum()
    if denominator.item() <= 0:
        return logits.sum() * 0
    return (token_loss * effective_weight).sum() / denominator


def _weighted_mean(
    values: List[Tuple[torch.Tensor, int]], reference: torch.Tensor
) -> torch.Tensor:
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return reference.sum() * 0
    return sum(value * weight for value, weight in values) / total_weight


def _parent_states(
    box_states: torch.Tensor,
    indices: torch.Tensor,
    task_states: torch.Tensor,
    task_name: str,
) -> torch.Tensor:
    indices = indices.to(device=box_states.device, dtype=torch.long)
    if indices.numel() != task_states.shape[0]:
        raise RuntimeError(
            f"{task_name} person binding count mismatch: "
            f"indices={indices.numel()} states={task_states.shape[0]}"
        )
    if indices.numel() and (
        int(indices.min().item()) < 0
        or int(indices.max().item()) >= box_states.shape[0]
    ):
        raise RuntimeError(
            f"{task_name} person binding index is outside box records"
        )
    return box_states.index_select(0, indices)


class Stage1V1Model(nn.Module):
    def __init__(
        self,
        *,
        base_model: nn.Module,
        hidden_size: int,
        token_ids: Dict[str, int],
        image_token_id: int,
        decoder_dim: int = 256,
        decoder_heads: int = 8,
        pose_decoder_type: str = "direct_joint_query",
        visual_merge_size: int = 2,
        loss_weights: Optional[Dict[str, float]] = None,
        task_backbone_grad_scale: float = 0.1,
        pose_box_jitter_enabled: bool = False,
        pose_box_shift_factor: float = 0.16,
        pose_box_shift_prob: float = 0.3,
        pose_box_scale_range: Tuple[float, float] = (0.5, 1.5),
        pose_box_scale_prob: float = 1.0,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.token_ids = {name: int(value) for name, value in token_ids.items()}
        self.image_token_id = int(image_token_id)
        self.visual_merge_size = int(visual_merge_size)
        self.task_backbone_grad_scale = float(task_backbone_grad_scale)
        if not 0.0 <= self.task_backbone_grad_scale <= 1.0:
            raise ValueError("task backbone gradient scale must be in [0, 1]")
        self.pose_box_jitter_enabled = bool(pose_box_jitter_enabled)
        self.pose_box_shift_factor = float(pose_box_shift_factor)
        self.pose_box_shift_prob = float(pose_box_shift_prob)
        self.pose_box_scale_range = tuple(
            float(value) for value in pose_box_scale_range
        )
        self.pose_box_scale_prob = float(pose_box_scale_prob)
        jitter_pose_boxes(
            torch.empty((0, 4)),
            shift_factor=self.pose_box_shift_factor,
            shift_prob=self.pose_box_shift_prob,
            scale_range=self.pose_box_scale_range,
            scale_prob=self.pose_box_scale_prob,
        )
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
        if loss_weights:
            self.loss_weights.update(
                {name: float(value) for name, value in loss_weights.items()}
            )
        self.box_head = BoxQualityHead(hidden_size, decoder_dim)
        self.pose_decoder_type = str(pose_decoder_type)
        if self.pose_decoder_type == "direct_joint_query":
            self.pose_decoder = DirectJointQueryPoseDecoder(
                hidden_size,
                hidden_size,
                decoder_dim=decoder_dim,
                num_heads=decoder_heads,
            )
        elif self.pose_decoder_type == "topdown_heatmap":
            self.pose_decoder = TopDownHeatmapPoseDecoder(
                hidden_size,
                hidden_size,
                decoder_dim=decoder_dim,
            )
        elif self.pose_decoder_type == "coordinate":
            self.pose_decoder = MultiPersonPoseDecoder(
                hidden_size,
                hidden_size,
                decoder_dim=decoder_dim,
                num_heads=decoder_heads,
            )
        else:
            raise ValueError(
                "pose_decoder_type must be direct_joint_query, topdown_heatmap, or coordinate"
            )
        self.mask_decoder = PersonMaskDecoder(
            hidden_size, hidden_size, decoder_dim=decoder_dim
        )
        self.pose_roi_enabled = self.pose_decoder_type != "direct_joint_query"
        self.pose_joint_reference_enabled = self.pose_decoder_type == "coordinate"
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
            raise RuntimeError("Stage1 image training requires temporal grid 1")
        if (
            grid_height % self.visual_merge_size
            or grid_width % self.visual_merge_size
        ):
            raise RuntimeError("visual grid is not divisible by merge size")
        height = grid_height // self.visual_merge_size
        width = grid_width // self.visual_merge_size
        expected = height * width
        if visual.shape[0] != expected:
            raise RuntimeError(
                "visual token/grid mismatch: "
                f"tokens={visual.shape[0]} expected={expected} "
                f"grid={temporal, grid_height, grid_width}"
            )
        return visual, (height, width)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        language_loss_weights: Optional[torch.Tensor] = None,
        box_targets: Sequence[torch.Tensor],
        pose_targets: Sequence[torch.Tensor],
        pose_areas: Sequence[torch.Tensor],
        seg_targets: Sequence[torch.Tensor],
        box_person_indices: Sequence[torch.Tensor],
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
        reference = hidden
        task_hidden = scale_task_gradient(
            hidden, self.task_backbone_grad_scale
        )
        ce_loss = weighted_causal_language_loss(
            outputs.logits, labels, language_loss_weights
        )
        batch_size = input_ids.shape[0]
        if not all(
            len(values) == batch_size
            for values in [
                box_targets,
                pose_targets,
                pose_areas,
                seg_targets,
                box_person_indices,
                pose_person_indices,
                seg_person_indices,
                original_sizes,
            ]
        ):
            raise ValueError("task target lists must match batch size")

        box_states_by_image = require_exact_token_states(
            input_ids=input_ids,
            hidden=task_hidden,
            labels=labels,
            token_id=self.token_ids["box"],
            expected_counts=[len(value) for value in box_targets],
            task_name="box",
        )
        pose_states_by_image = require_exact_token_states(
            input_ids=input_ids,
            hidden=task_hidden,
            labels=labels,
            token_id=self.token_ids["pose"],
            expected_counts=[len(value) for value in pose_targets],
            task_name="pose",
        )
        seg_states_by_image = require_exact_token_states(
            input_ids=input_ids,
            hidden=task_hidden,
            labels=labels,
            token_id=self.token_ids["seg"],
            expected_counts=[len(value) for value in seg_targets],
            task_name="seg",
        )

        box_l1_values: List[Tuple[torch.Tensor, int]] = []
        box_giou_values: List[Tuple[torch.Tensor, int]] = []
        quality_values: List[Tuple[torch.Tensor, int]] = []
        pose_l1_values: List[Tuple[torch.Tensor, int]] = []
        pose_oks_values: List[Tuple[torch.Tensor, int]] = []
        pose_heatmap_values: List[Tuple[torch.Tensor, int]] = []
        pose_visibility_values: List[Tuple[torch.Tensor, int]] = []
        mask_bce_values: List[Tuple[torch.Tensor, int]] = []
        mask_dice_values: List[Tuple[torch.Tensor, int]] = []
        predictions: List[Dict[str, torch.Tensor]] = []

        for batch_index in range(batch_size):
            box_states = box_states_by_image[batch_index]
            pose_states = pose_states_by_image[batch_index]
            seg_states = seg_states_by_image[batch_index]
            visual_tokens, spatial_shape = self._visual_tokens_and_shape(
                input_ids, task_hidden, image_grid_thw, batch_index
            )
            box_target = box_targets[batch_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            predicted_boxes, predicted_quality_logits = self.box_head(box_states)
            box_count = int(box_target.shape[0])
            if box_count:
                box_l1_values.append(
                    (F.l1_loss(predicted_boxes, box_target), box_count)
                )
                box_giou_values.append(
                    (
                        generalized_box_iou_loss(predicted_boxes, box_target),
                        box_count,
                    )
                )
                quality_target = aligned_box_iou(
                    predicted_boxes.detach(), box_target
                )
                quality_values.append(
                    (
                        F.binary_cross_entropy_with_logits(
                            predicted_quality_logits, quality_target
                        ),
                        box_count,
                    )
                )

            pose_parent = _parent_states(
                box_states,
                pose_person_indices[batch_index],
                pose_states,
                "pose",
            )
            pose_indices = pose_person_indices[batch_index].to(
                device=hidden.device, dtype=torch.long
            )
            if self.pose_decoder_type == "direct_joint_query":
                pose_reference_boxes = select_pose_reference_boxes(
                    predicted_boxes,
                    box_target,
                    training=self.training,
                )
                pose_boxes = pose_reference_boxes.index_select(0, pose_indices)
                pose_boxes_were_jittered = False
                pose_visual_tokens = visual_tokens.unsqueeze(0)
                pose_box_condition = pose_boxes
            else:
                pose_boxes = box_target.index_select(0, pose_indices)
                pose_boxes_were_jittered = (
                    self.training
                    and self.pose_box_jitter_enabled
                    and self.pose_decoder_type == "topdown_heatmap"
                )
                if pose_boxes_were_jittered:
                    pose_boxes = jitter_pose_boxes(
                        pose_boxes,
                        shift_factor=self.pose_box_shift_factor,
                        shift_prob=self.pose_box_shift_prob,
                        scale_range=self.pose_box_scale_range,
                        scale_prob=self.pose_box_scale_prob,
                    )
                if self.pose_roi_enabled:
                    pose_visual_tokens = roi_pool_visual_tokens(
                        visual_tokens,
                        spatial_shape=spatial_shape,
                        boxes=pose_boxes,
                        output_size=7,
                    )
                    pose_box_condition = pose_boxes
                else:
                    pose_visual_tokens = visual_tokens.unsqueeze(0)
                    pose_box_condition = None
            predicted_heatmaps = None
            if self.pose_decoder_type == "direct_joint_query":
                predicted_pose, predicted_visibility = self.pose_decoder(
                    pose_states,
                    pose_parent,
                    pose_visual_tokens,
                    spatial_shape=spatial_shape,
                    person_boxes=pose_box_condition,
                )
            elif self.pose_decoder_type == "topdown_heatmap":
                if pose_box_condition is None:
                    raise RuntimeError("topdown heatmap pose requires ROI boxes")
                (
                    predicted_pose,
                    predicted_visibility,
                    predicted_heatmaps,
                ) = self.pose_decoder(
                    pose_states,
                    pose_parent,
                    pose_visual_tokens,
                    person_boxes=pose_box_condition,
                )
            else:
                predicted_pose, predicted_visibility = self.pose_decoder(
                    pose_states,
                    pose_parent,
                    pose_visual_tokens,
                    person_boxes=pose_box_condition,
                    use_joint_reference=self.pose_joint_reference_enabled,
                )
            pose_target = pose_targets[batch_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            area_target = pose_areas[batch_index].to(
                device=hidden.device, dtype=hidden.dtype
            )
            pose_count = int(pose_target.shape[0])
            if pose_count:
                target_visibility = pose_target[..., 2].long().clamp(0, 2)
                if pose_boxes_were_jittered:
                    inside_pose_roi = keypoints_inside_boxes(
                        pose_target[..., :2], pose_boxes
                    )
                    target_visibility = target_visibility.masked_fill(
                        ~inside_pose_roi, 0
                    )
                visible = target_visibility > 0
                if predicted_heatmaps is not None:
                    pose_heatmap_values.append(
                        (
                            keypoint_heatmap_mse_loss(
                                predicted_heatmaps,
                                pose_target[..., :2],
                                target_visibility,
                                pose_boxes,
                            ),
                            pose_count,
                        )
                    )
                if visible.any():
                    if self.pose_decoder_type == "direct_joint_query":
                        pose_l1 = F.l1_loss(
                            predicted_pose[visible],
                            pose_target[..., :2][visible],
                        )
                    else:
                        pose_l1 = roi_relative_pose_l1(
                            predicted_pose,
                            pose_target[..., :2],
                            target_visibility,
                            pose_boxes,
                        )
                    pose_l1_values.append(
                        (
                            pose_l1,
                            int(visible.sum().item()) * 2,
                        )
                    )
                pose_visibility_values.append(
                    (
                        F.cross_entropy(
                            predicted_visibility.reshape(-1, 3),
                            target_visibility.reshape(-1),
                        ),
                        pose_count * 17,
                    )
                )
                size = torch.tensor(
                    [original_sizes[batch_index]],
                    device=hidden.device,
                    dtype=hidden.dtype,
                ).expand(pose_count, -1)
                pose_oks_values.append(
                    (
                        oks_loss(
                            predicted_pose,
                            pose_target[..., :2],
                            target_visibility,
                            area_target,
                            size,
                        ),
                        pose_count,
                    )
                )

            seg_parent = _parent_states(
                box_states,
                seg_person_indices[batch_index],
                seg_states,
                "seg",
            )
            predicted_masks = self.mask_decoder(
                seg_states,
                seg_parent,
                visual_tokens.unsqueeze(0),
                spatial_shape=spatial_shape,
            )
            seg_target = seg_targets[batch_index].to(device=hidden.device)
            seg_count = int(seg_target.shape[0])
            if seg_count:
                bce, dice = mask_bce_dice_loss(predicted_masks, seg_target)
                mask_bce_values.append((bce, seg_count))
                mask_dice_values.append((dice, seg_count))
            if return_predictions:
                predictions.append(
                    {
                        "boxes": predicted_boxes,
                        "box_quality": torch.sigmoid(predicted_quality_logits),
                        "pose_xy": predicted_pose,
                        "pose_visibility": predicted_visibility,
                        "mask_logits": predicted_masks,
                    }
                )

        raw_losses = {
            "ce_loss": ce_loss,
            "box_l1": _weighted_mean(box_l1_values, reference),
            "box_giou": _weighted_mean(box_giou_values, reference),
            "box_quality": _weighted_mean(quality_values, reference),
            "pose_l1": _weighted_mean(pose_l1_values, reference),
            "pose_oks": _weighted_mean(pose_oks_values, reference),
            "pose_heatmap": _weighted_mean(pose_heatmap_values, reference),
            "pose_visibility": _weighted_mean(
                pose_visibility_values, reference
            ),
            "mask_bce": _weighted_mean(mask_bce_values, reference),
            "mask_dice": _weighted_mean(mask_dice_values, reference),
        }
        total = raw_losses["ce_loss"] * self.loss_weights["ce"]
        result: Dict[str, Any] = dict(raw_losses)
        for name in [
            "box_l1",
            "box_giou",
            "box_quality",
            "pose_l1",
            "pose_oks",
            "pose_heatmap",
            "pose_visibility",
            "mask_bce",
            "mask_dice",
        ]:
            weighted = raw_losses[name] * self.loss_weights[name]
            result[f"weighted_{name}"] = weighted
            total = total + weighted
        result["loss"] = total
        result["valid_box_count"] = input_ids.new_tensor(
            sum(len(value) for value in box_targets)
        )
        result["valid_pose_count"] = input_ids.new_tensor(
            sum(len(value) for value in pose_targets)
        )
        result["valid_seg_count"] = input_ids.new_tensor(
            sum(len(value) for value in seg_targets)
        )
        if return_predictions:
            result["predictions"] = predictions
        return result

    def save_task_heads(self, directory: str) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        state = {
            "box_head": self.box_head.state_dict(),
            "pose_decoder": self.pose_decoder.state_dict(),
            "pose_decoder_type": self.pose_decoder_type,
            "mask_decoder": self.mask_decoder.state_dict(),
            "token_ids": self.token_ids,
            "image_token_id": self.image_token_id,
            "visual_merge_size": self.visual_merge_size,
            "loss_weights": self.loss_weights,
            "pose_roi_enabled": self.pose_roi_enabled,
            "pose_joint_reference_enabled": self.pose_joint_reference_enabled,
            "pose_box_jitter": {
                "enabled": self.pose_box_jitter_enabled,
                "shift_factor": self.pose_box_shift_factor,
                "shift_prob": self.pose_box_shift_prob,
                "scale_range": list(self.pose_box_scale_range),
                "scale_prob": self.pose_box_scale_prob,
            },
            "task_backbone_grad_scale": self.task_backbone_grad_scale,
        }
        torch.save(state, output / "stage1_task_heads.pt")


    def load_task_heads(
        self,
        path: str,
        *,
        restore_training_config: bool = True,
        load_pose_decoder: bool = True,
    ) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state["token_ids"] != self.token_ids:
            raise RuntimeError("checkpoint special-token ids do not match")
        if int(state["image_token_id"]) != self.image_token_id:
            raise RuntimeError("checkpoint image token id does not match")
        if int(state["visual_merge_size"]) != self.visual_merge_size:
            raise RuntimeError("checkpoint visual merge size does not match")
        checkpoint_pose_type = str(
            state.get("pose_decoder_type", "coordinate")
        )
        if checkpoint_pose_type != self.pose_decoder_type:
            raise RuntimeError(
                "pose decoder type mismatch: "
                f"checkpoint={checkpoint_pose_type} "
                f"model={self.pose_decoder_type}"
            )
        self.box_head.load_state_dict(state["box_head"], strict=True)
        legacy_pose = "pose_roi_enabled" not in state
        legacy_joint_reference = "pose_joint_reference_enabled" not in state
        self.pose_joint_reference_enabled = bool(
            state.get("pose_joint_reference_enabled", False)
        )
        self.pose_roi_enabled = bool(state.get("pose_roi_enabled", False))
        if restore_training_config:
            self.task_backbone_grad_scale = float(
                state.get("task_backbone_grad_scale", self.task_backbone_grad_scale)
            )
        jitter_state = state.get("pose_box_jitter")
        if restore_training_config and jitter_state is not None:
            self.pose_box_jitter_enabled = bool(jitter_state["enabled"])
            self.pose_box_shift_factor = float(jitter_state["shift_factor"])
            self.pose_box_shift_prob = float(jitter_state["shift_prob"])
            self.pose_box_scale_range = tuple(
                float(value) for value in jitter_state["scale_range"]
            )
            self.pose_box_scale_prob = float(jitter_state["scale_prob"])
            jitter_pose_boxes(
                torch.empty((0, 4)),
                shift_factor=self.pose_box_shift_factor,
                shift_prob=self.pose_box_shift_prob,
                scale_range=self.pose_box_scale_range,
                scale_prob=self.pose_box_scale_prob,
            )
        if not load_pose_decoder:
            pass
        elif self.pose_decoder_type == "coordinate":
            incompatible = self.pose_decoder.load_state_dict(
                state["pose_decoder"], strict=False
            )
            allowed_missing = set()
            if legacy_pose:
                allowed_missing.update(
                    {"box_project.weight", "box_project.bias"}
                )
            if legacy_joint_reference:
                allowed_missing.add("joint_reference_logits")
            unexpected_missing = (
                set(incompatible.missing_keys) - allowed_missing
            )
            if unexpected_missing or incompatible.unexpected_keys:
                raise RuntimeError(
                    "pose decoder checkpoint mismatch: "
                    f"missing={sorted(unexpected_missing)} "
                    f"unexpected={sorted(incompatible.unexpected_keys)}"
                )
        else:
            self.pose_decoder.load_state_dict(
                state["pose_decoder"], strict=True
            )
        self.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        if restore_training_config:
            self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS)
            self.loss_weights.update(
                {
                    name: float(value)
                    for name, value in state["loss_weights"].items()
                }
            )
