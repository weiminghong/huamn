from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from human_lisa_qwen_v2.heads_v2 import (
    MultiScaleVisualNeck,
    mask_bce_dice_loss,
    oks_loss,
    require_exact_token_states,
)
from human_lisa_qwen_v2.modeling_stage1_v2 import (
    _find_vision_module,
    _raster_indices,
    _weighted_mean,
    causal_language_losses,
)
from human_lisa_qwen_v3_boxpose.heads_v3 import (
    QueryConditionedMaskDecoder,
    QueryConditionedPoseDecoder,
    SetPersonQueryDecoder,
    box_cxcywh_to_xyxy,
    match_person_queries,
    person_query_losses,
)


DEFAULT_LOSS_WEIGHTS_V3 = {
    "ce": 1.0,
    "query_objectness": 1.0,
    "query_bbox_l1": 5.0,
    "query_giou": 2.0,
    "query_aux": 0.5,
    "pose_l1": 10.0,
    "pose_oks": 4.0,
    "pose_visibility": 1.0,
    "pose_aux": 0.5,
    "mask_bce": 2.0,
    "mask_dice": 5.0,
}


class Stage1V3Model(nn.Module):
    """Hybrid structured VLM with an image-conditioned K-person query set."""

    def __init__(
        self,
        *,
        base_model: nn.Module,
        text_hidden_size: int,
        token_ids: Dict[str, int],
        vision_layer_indices: Sequence[int] = (8, 16, 24),
        decoder_dim: int = 256,
        decoder_heads: int = 8,
        person_query_layers: int = 3,
        pose_decoder_layers: int = 2,
        visual_merge_size: int = 2,
        max_people: int = 25,
        loss_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.token_ids = {name: int(value) for name, value in token_ids.items()}
        self.vision_layer_indices = tuple(int(value) for value in vision_layer_indices)
        self.visual_merge_size = int(visual_merge_size)
        self.decoder_dim = int(decoder_dim)
        self.decoder_heads = int(decoder_heads)
        self.person_query_layers = int(person_query_layers)
        self.pose_decoder_layers = int(pose_decoder_layers)
        self.max_people = int(max_people)
        self.loss_weights = dict(DEFAULT_LOSS_WEIGHTS_V3)
        if loss_weights:
            self.loss_weights.update({name: float(value) for name, value in loss_weights.items()})
        vision_module = _find_vision_module(base_model)
        object.__setattr__(self, "_vision_module_ref", vision_module)
        vision_hidden_size = int(vision_module.config.hidden_size)
        self.visual_neck = MultiScaleVisualNeck(
            vision_hidden_size,
            len(self.vision_layer_indices),
            decoder_dim,
        )
        self.person_query_decoder = SetPersonQueryDecoder(
            decoder_dim=decoder_dim,
            num_heads=decoder_heads,
            num_layers=person_query_layers,
            num_queries=max_people,
            num_feature_levels=3,
        )
        self.pose_decoder = QueryConditionedPoseDecoder(
            text_hidden_size,
            decoder_dim=decoder_dim,
            num_layers=pose_decoder_layers,
        )
        self.mask_decoder = QueryConditionedMaskDecoder(
            text_hidden_size,
            decoder_dim=decoder_dim,
            num_heads=decoder_heads,
        )
        self._captured_vision_output: Any = None
        self._vision_hook_handle = vision_module.register_forward_hook(
            self._capture_vision_output,
            with_kwargs=True,
        )
        if hasattr(self.base_model, "config"):
            self.base_model.config.use_cache = False

    def _capture_vision_output(self, module: nn.Module, args: Any, kwargs: Any, output: Any) -> None:
        del module, args, kwargs
        self._captured_vision_output = output

    def _forward_backbone(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        logit_positions: Optional[torch.Tensor] = None,
        **base_kwargs: Any,
    ) -> Tuple[Any, torch.Tensor, List[Tuple[List[torch.Tensor], torch.Tensor]]]:
        self._captured_vision_output = None
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            labels=None,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            logits_to_keep=1 if logit_positions is None else logit_positions,
            **base_kwargs,
        )
        vision_output = self._captured_vision_output
        self._captured_vision_output = None
        if vision_output is None or getattr(vision_output, "hidden_states", None) is None:
            raise RuntimeError("Qwen vision tower did not expose hidden states")
        all_layers = vision_output.hidden_states
        if max(self.vision_layer_indices) >= len(all_layers):
            raise RuntimeError(
                f"vision layer request {self.vision_layer_indices} exceeds {len(all_layers)} states"
            )
        selected = [all_layers[index] for index in self.vision_layer_indices]
        feature_maps: List[Tuple[List[torch.Tensor], torch.Tensor]] = []
        offset = 0
        for grid in image_grid_thw:
            temporal, height, width = [int(value) for value in grid.detach().cpu().tolist()]
            if temporal != 1:
                raise RuntimeError("Stage1 V3 accepts still images only")
            count = temporal * height * width
            layers = [value[offset : offset + count] for value in selected]
            raster = _raster_indices(height, width, self.visual_merge_size, layers[0].device)
            feature_maps.append(
                self.visual_neck(layers, raster_indices=raster, spatial_shape=(height, width))
            )
            offset += count
        if offset != int(selected[0].shape[0]):
            raise RuntimeError("vision feature split did not consume every raw patch token")
        return outputs, outputs.hidden_states[-1], feature_maps

    @staticmethod
    def _target_to_query(
        *,
        target_count: int,
        query_indices: torch.Tensor,
        target_indices: torch.Tensor,
    ) -> torch.Tensor:
        if target_count == 0:
            return query_indices.new_empty((0,))
        if query_indices.numel() != target_count or target_indices.numel() != target_count:
            raise RuntimeError("Hungarian matcher did not cover every GT person")
        target_to_query = torch.empty(target_count, device=query_indices.device, dtype=torch.long)
        target_to_query[target_indices] = query_indices
        return target_to_query

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: torch.Tensor,
        image_grid_thw: torch.Tensor,
        language_loss_weights: Optional[torch.Tensor],
        language_task_ids: Optional[torch.Tensor],
        num_persons: Sequence[int],
        box_targets: Sequence[torch.Tensor],
        pose_targets: Sequence[torch.Tensor],
        pose_areas: Sequence[torch.Tensor],
        seg_targets: Sequence[torch.Tensor],
        pose_person_indices: Sequence[torch.Tensor],
        seg_person_indices: Sequence[torch.Tensor],
        original_sizes: Sequence[Tuple[int, int]],
        return_predictions: bool = False,
        **base_kwargs: Any,
    ) -> Dict[str, Any]:
        logit_positions = labels[:, 1:].ne(-100).any(dim=0).nonzero(as_tuple=False).flatten()
        if logit_positions.numel() == 0:
            raise RuntimeError("batch has no supervised assistant token")
        outputs, hidden, visual_features = self._forward_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            logit_positions=logit_positions,
            **base_kwargs,
        )
        language = causal_language_losses(
            outputs.logits,
            labels,
            language_loss_weights,
            language_task_ids,
            logit_positions,
        )
        people_states = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=self.token_ids["person"],
            expected_counts=num_persons,
            task_name="person",
        )
        pose_states_all = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=self.token_ids["pose"],
            expected_counts=num_persons,
            task_name="pose",
        )
        seg_states_all = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=self.token_ids["seg"],
            expected_counts=num_persons,
            task_name="seg",
        )
        batch_size = int(input_ids.shape[0])
        values = [
            num_persons,
            box_targets,
            pose_targets,
            pose_areas,
            seg_targets,
            pose_person_indices,
            seg_person_indices,
            original_sizes,
            visual_features,
        ]
        if not all(len(value) == batch_size for value in values):
            raise ValueError("task target lists must match batch size")

        query_objectness_values: List[Tuple[torch.Tensor, int]] = []
        query_bbox_values: List[Tuple[torch.Tensor, int]] = []
        query_giou_values: List[Tuple[torch.Tensor, int]] = []
        query_aux_values: List[Tuple[torch.Tensor, int]] = []
        pose_l1_values: List[Tuple[torch.Tensor, int]] = []
        pose_oks_values: List[Tuple[torch.Tensor, int]] = []
        pose_vis_values: List[Tuple[torch.Tensor, int]] = []
        pose_aux_values: List[Tuple[torch.Tensor, int]] = []
        mask_bce_values: List[Tuple[torch.Tensor, int]] = []
        mask_dice_values: List[Tuple[torch.Tensor, int]] = []
        predictions: List[Dict[str, torch.Tensor]] = []

        for batch_index in range(batch_size):
            target_boxes = box_targets[batch_index].to(device=hidden.device, dtype=hidden.dtype)
            target_count = int(target_boxes.shape[0])
            if target_count != int(num_persons[batch_index]):
                raise RuntimeError("person count and box target count differ")
            pyramid, mask_features = visual_features[batch_index]
            query_features, objectness_logits, boxes_cxcywh, auxiliary_queries = self.person_query_decoder(pyramid)
            query_indices, target_indices = match_person_queries(
                objectness_logits,
                boxes_cxcywh,
                target_boxes,
            )
            query_loss = person_query_losses(
                objectness_logits,
                boxes_cxcywh,
                target_boxes,
                query_indices,
                target_indices,
            )
            query_objectness_values.append((query_loss["query_objectness"], max(1, target_count)))
            query_bbox_values.append((query_loss["query_bbox_l1"], max(1, target_count)))
            query_giou_values.append((query_loss["query_giou"], max(1, target_count)))
            if auxiliary_queries:
                auxiliary_total = hidden.sum() * 0
                for auxiliary in auxiliary_queries:
                    aux_query_indices, aux_target_indices = match_person_queries(
                        auxiliary["objectness_logits"],
                        auxiliary["boxes_cxcywh"],
                        target_boxes,
                    )
                    auxiliary_loss = person_query_losses(
                        auxiliary["objectness_logits"],
                        auxiliary["boxes_cxcywh"],
                        target_boxes,
                        aux_query_indices,
                        aux_target_indices,
                    )
                    auxiliary_total = auxiliary_total + (
                        auxiliary_loss["query_objectness"] * self.loss_weights["query_objectness"]
                        + auxiliary_loss["query_bbox_l1"] * self.loss_weights["query_bbox_l1"]
                        + auxiliary_loss["query_giou"] * self.loss_weights["query_giou"]
                    )
                query_aux_values.append((auxiliary_total / len(auxiliary_queries), max(1, target_count)))

            target_to_query = self._target_to_query(
                target_count=target_count,
                query_indices=query_indices,
                target_indices=target_indices,
            )
            ordered_query_features = query_features.index_select(0, target_to_query)
            ordered_query_boxes = boxes_cxcywh.index_select(0, target_to_query)

            pose_indices = pose_person_indices[batch_index].to(hidden.device, dtype=torch.long)
            pose_query_features = ordered_query_features.index_select(0, pose_indices)
            pose_query_boxes = ordered_query_boxes.index_select(0, pose_indices)
            person_pose = people_states[batch_index].index_select(0, pose_indices)
            pose_states = pose_states_all[batch_index].index_select(0, pose_indices)
            predicted_pose, predicted_visibility, auxiliary_pose = self.pose_decoder(
                pose_query_features,
                person_pose,
                pose_states,
                pose_query_boxes,
                pyramid,
            )
            pose_target = pose_targets[batch_index].to(device=hidden.device, dtype=hidden.dtype)
            area_target = pose_areas[batch_index].to(device=hidden.device, dtype=hidden.dtype)
            pose_count = int(pose_target.shape[0])
            if pose_count:
                target_visibility = pose_target[..., 2].long().clamp(0, 2)
                visible = target_visibility > 0
                if visible.any():
                    visible_count = int(visible.sum().item())
                    pose_l1_values.append(
                        (
                            F.l1_loss(predicted_pose[visible], pose_target[..., :2][visible]),
                            visible_count * 2,
                        )
                    )
                    for auxiliary in auxiliary_pose:
                        pose_aux_values.append(
                            (
                                F.l1_loss(auxiliary[visible], pose_target[..., :2][visible]),
                                visible_count * 2,
                            )
                        )
                pose_vis_values.append(
                    (
                        F.cross_entropy(
                            predicted_visibility.reshape(-1, 3),
                            target_visibility.reshape(-1),
                        ),
                        pose_count * 17,
                    )
                )
                image_size = torch.tensor(
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
                            image_size,
                        ),
                        pose_count,
                    )
                )

            seg_indices = seg_person_indices[batch_index].to(hidden.device, dtype=torch.long)
            seg_query_features = ordered_query_features.index_select(0, seg_indices)
            person_seg = people_states[batch_index].index_select(0, seg_indices)
            seg_states = seg_states_all[batch_index].index_select(0, seg_indices)
            predicted_masks = self.mask_decoder(
                seg_query_features,
                person_seg,
                seg_states,
                pyramid[0],
                mask_features,
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
                        "pose_xy": predicted_pose,
                        "pose_visibility": predicted_visibility,
                        "mask_logits": predicted_masks,
                        "pose_person_indices": pose_indices,
                        "seg_person_indices": seg_indices,
                        "matched_query_boxes_xyxy": box_cxcywh_to_xyxy(ordered_query_boxes),
                        "matched_query_objectness": objectness_logits.index_select(0, target_to_query),
                        "target_to_query": target_to_query,
                    }
                )

        reference = hidden
        raw_losses: Dict[str, torch.Tensor] = {
            **language,
            "query_objectness": _weighted_mean(query_objectness_values, reference),
            "query_bbox_l1": _weighted_mean(query_bbox_values, reference),
            "query_giou": _weighted_mean(query_giou_values, reference),
            "query_aux": _weighted_mean(query_aux_values, reference),
            "pose_l1": _weighted_mean(pose_l1_values, reference),
            "pose_oks": _weighted_mean(pose_oks_values, reference),
            "pose_visibility": _weighted_mean(pose_vis_values, reference),
            "pose_aux": _weighted_mean(pose_aux_values, reference),
            "mask_bce": _weighted_mean(mask_bce_values, reference),
            "mask_dice": _weighted_mean(mask_dice_values, reference),
        }
        total = raw_losses["ce_loss"] * self.loss_weights["ce"]
        result: Dict[str, Any] = dict(raw_losses)
        weighted_names = [
            "query_objectness",
            "query_bbox_l1",
            "query_giou",
            "query_aux",
            "pose_l1",
            "pose_oks",
            "pose_visibility",
            "pose_aux",
            "mask_bce",
            "mask_dice",
        ]
        for name in weighted_names:
            weighted = raw_losses[name] * self.loss_weights[name]
            result[f"weighted_{name}"] = weighted
            total = total + weighted
        result["loss"] = total
        result["valid_pose_count"] = input_ids.new_tensor(sum(len(value) for value in pose_targets))
        result["valid_seg_count"] = input_ids.new_tensor(sum(len(value) for value in seg_targets))
        result["person_count"] = input_ids.new_tensor(sum(int(value) for value in num_persons))
        if return_predictions:
            result["predictions"] = predictions
        return result

    @torch.no_grad()
    def decode_dense_from_inputs(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw: torch.Tensor,
        target_start: int,
        person_count: int,
        generated_boxes_xyxy: torch.Tensor,
        **base_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        outputs, hidden, visual_features = self._forward_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            **base_kwargs,
        )
        del outputs
        if input_ids.shape[0] != 1 or len(visual_features) != 1:
            raise ValueError("dense generated decoding currently accepts one image")
        if generated_boxes_xyxy.shape != (int(person_count), 4):
            raise ValueError("generated box count does not match generated person count")
        counts = [int(person_count)]
        person = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=None,
            token_id=self.token_ids["person"],
            expected_counts=counts,
            task_name="person",
            minimum_position=target_start,
        )[0]
        pose = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=None,
            token_id=self.token_ids["pose"],
            expected_counts=counts,
            task_name="pose",
            minimum_position=target_start,
        )[0]
        seg = require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=None,
            token_id=self.token_ids["seg"],
            expected_counts=counts,
            task_name="seg",
            minimum_position=target_start,
        )[0]
        pyramid, mask_features = visual_features[0]
        query_features, objectness_logits, boxes_cxcywh, _ = self.person_query_decoder(pyramid)
        query_indices, target_indices = match_person_queries(
            objectness_logits,
            boxes_cxcywh,
            generated_boxes_xyxy.to(device=boxes_cxcywh.device, dtype=boxes_cxcywh.dtype),
        )
        target_to_query = self._target_to_query(
            target_count=int(person_count),
            query_indices=query_indices,
            target_indices=target_indices,
        )
        matched_features = query_features.index_select(0, target_to_query)
        matched_boxes = boxes_cxcywh.index_select(0, target_to_query)
        pose_xy, pose_visibility, _ = self.pose_decoder(
            matched_features,
            person,
            pose,
            matched_boxes,
            pyramid,
        )
        mask_logits = self.mask_decoder(
            matched_features,
            person,
            seg,
            pyramid[0],
            mask_features,
        )
        person_positions = (
            input_ids[0].eq(self.token_ids["person"])
            & torch.arange(input_ids.shape[1], device=input_ids.device).ge(int(target_start))
        ).nonzero(as_tuple=False).flatten()
        preceding = (person_positions - 1).clamp(min=0)
        logits = self.base_model.get_output_embeddings()(hidden[0].index_select(0, preceding)).float()
        person_probability = logits.softmax(dim=-1)[:, self.token_ids["person"]]
        return {
            "pose_xy": pose_xy,
            "pose_visibility_logits": pose_visibility,
            "mask_logits": mask_logits,
            "person_probability": person_probability,
            "query_boxes_xyxy": box_cxcywh_to_xyxy(matched_boxes),
            "query_objectness": objectness_logits.index_select(0, target_to_query).sigmoid(),
        }

    def save_task_modules(self, directory: str) -> None:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "visual_neck": self.visual_neck.state_dict(),
                "person_query_decoder": self.person_query_decoder.state_dict(),
                "pose_decoder": self.pose_decoder.state_dict(),
                "mask_decoder": self.mask_decoder.state_dict(),
                "token_ids": self.token_ids,
                "vision_layer_indices": list(self.vision_layer_indices),
                "visual_merge_size": self.visual_merge_size,
                "loss_weights": self.loss_weights,
                "decoder_dim": self.decoder_dim,
                "decoder_heads": self.decoder_heads,
                "person_query_layers": self.person_query_layers,
                "pose_decoder_layers": self.pose_decoder_layers,
                "max_people": self.max_people,
            },
            output / "stage1_v3_task_modules.pt",
        )

    def load_task_modules(self, path: str) -> None:
        state = torch.load(path, map_location="cpu", weights_only=False)
        checks = {
            "token_ids": state["token_ids"] == self.token_ids,
            "vision_layer_indices": tuple(state["vision_layer_indices"]) == self.vision_layer_indices,
            "visual_merge_size": int(state["visual_merge_size"]) == self.visual_merge_size,
            "decoder_dim": int(state["decoder_dim"]) == self.decoder_dim,
            "decoder_heads": int(state["decoder_heads"]) == self.decoder_heads,
            "person_query_layers": int(state["person_query_layers"]) == self.person_query_layers,
            "pose_decoder_layers": int(state["pose_decoder_layers"]) == self.pose_decoder_layers,
            "max_people": int(state["max_people"]) == self.max_people,
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            raise RuntimeError(f"checkpoint architecture mismatch: {failed}")
        self.visual_neck.load_state_dict(state["visual_neck"], strict=True)
        self.person_query_decoder.load_state_dict(state["person_query_decoder"], strict=True)
        self.pose_decoder.load_state_dict(state["pose_decoder"], strict=True)
        self.mask_decoder.load_state_dict(state["mask_decoder"], strict=True)
        self.loss_weights = {name: float(value) for name, value in state["loss_weights"].items()}
