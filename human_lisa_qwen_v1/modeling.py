from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen2_5_VLForConditionalGeneration = None

try:
    from transformers import AutoModelForVision2Seq
except Exception:  # pragma: no cover
    AutoModelForVision2Seq = None
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
except Exception:  # pragma: no cover
    Qwen3_5ForConditionalGeneration = None


def load_qwen_vl_backbone(model_name_or_path: str, torch_dtype: torch.dtype):
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    kwargs = dict(torch_dtype=torch_dtype, trust_remote_code=True, low_cpu_mem_usage=True)
    model_type = str(getattr(cfg, "model_type", "")).lower()
    if model_type == "qwen2_5_vl" and Qwen2_5_VLForConditionalGeneration is not None:
        return Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    if getattr(cfg, "vision_config", None) is not None and AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            pass
    if getattr(cfg, "vision_config", None) is not None and Qwen3_5ForConditionalGeneration is not None and model_type == "qwen3_5":
        return Qwen3_5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def apply_lora_if_requested(model, enable: bool, r: int = 16, alpha: int = 32, dropout: float = 0.05, target_modules: Any = None, modules_to_save: Any = None):
    if not enable:
        return model
    from peft import LoraConfig, get_peft_model
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if modules_to_save is None:
        modules_to_save = ["embed_tokens", "lm_head"]
    cfg = LoraConfig(
        r=int(r),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[..., 2] - boxes[..., 0]).clamp(min=0) * (boxes[..., 3] - boxes[..., 1]).clamp(min=0)


def generalized_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(pred[:, 0], target[:, 0])
    y1 = torch.maximum(pred[:, 1], target[:, 1])
    x2 = torch.minimum(pred[:, 2], target[:, 2])
    y2 = torch.minimum(pred[:, 3], target[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_p = box_area(pred)
    area_t = box_area(target)
    union = (area_p + area_t - inter).clamp(min=1e-7)
    iou = inter / union
    cx1 = torch.minimum(pred[:, 0], target[:, 0])
    cy1 = torch.minimum(pred[:, 1], target[:, 1])
    cx2 = torch.maximum(pred[:, 2], target[:, 2])
    cy2 = torch.maximum(pred[:, 3], target[:, 3])
    c_area = ((cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)).clamp(min=1e-7)
    giou = iou - (c_area - union) / c_area
    return (1.0 - giou).mean()


class TokenHeads(nn.Module):
    def __init__(self, hidden_size: int, seg_embedding_dim: int = 256):
        super().__init__()
        self.box_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 4),
        )
        self.pose_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 17 * 2 + 17 * 3),
        )
        self.seg_projector = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, seg_embedding_dim),
        )

    def decode_boxes(self, states: torch.Tensor) -> torch.Tensor:
        raw = torch.sigmoid(self.box_head(states))
        x1 = torch.minimum(raw[:, 0], raw[:, 2])
        x2 = torch.maximum(raw[:, 0], raw[:, 2])
        y1 = torch.minimum(raw[:, 1], raw[:, 3])
        y2 = torch.maximum(raw[:, 1], raw[:, 3])
        eps = 1e-4
        return torch.stack([x1, y1, torch.maximum(x2, x1 + eps), torch.maximum(y2, y1 + eps)], dim=-1).clamp(0, 1)

    def decode_pose(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raw = self.pose_head(states)
        xy = torch.sigmoid(raw[:, : 17 * 2]).view(-1, 17, 2)
        vis_logits = raw[:, 17 * 2 :].view(-1, 17, 3)
        return xy, vis_logits


class QwenLisaHeads(nn.Module):
    def __init__(self, base_model, hidden_size: int, token_ids: Dict[str, int], loss_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.base_model = base_model
        self.heads = TokenHeads(hidden_size)
        self.token_ids = token_ids
        self.loss_weights = {"ce": 1.0, "box_l1": 5.0, "box_giou": 2.0, "pose_xy": 10.0, "pose_vis": 1.0}
        if loss_weights:
            self.loss_weights.update({k: float(v) for k, v in loss_weights.items()})
        if hasattr(self.base_model, "config"):
            self.base_model.config.use_cache = False

    def _forward_base(self, input_ids, attention_mask=None, labels=None, **kwargs):
        base_kwargs = {k: v for k, v in kwargs.items() if v is not None}
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True,
            **base_kwargs,
        )

    def _token_states(self, input_ids: torch.Tensor, hidden: torch.Tensor, token_id: int, labels: Optional[torch.Tensor] = None, start_pos: Optional[int] = None) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        bsz = input_ids.shape[0]
        for b in range(bsz):
            mask = input_ids[b].eq(int(token_id))
            if labels is not None:
                mask = mask & labels[b].ne(-100)
            if start_pos is not None:
                pos_mask = torch.zeros_like(mask)
                pos_mask[int(start_pos):] = True
                mask = mask & pos_mask
            out.append(hidden[b][mask])
        return out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        box_targets: Optional[torch.Tensor] = None,
        box_target_mask: Optional[torch.Tensor] = None,
        pose_targets: Optional[torch.Tensor] = None,
        pose_target_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        outputs = self._forward_base(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)
        hidden = outputs.hidden_states[-1]
        loss = outputs.loss if outputs.loss is not None else hidden.sum() * 0
        logs: Dict[str, torch.Tensor] = {"ce_loss": loss.detach()}

        if box_targets is not None and box_target_mask is not None:
            states_by_b = self._token_states(input_ids, hidden, self.token_ids["box"], labels=labels)
            pred_list, tgt_list = [], []
            for b, states in enumerate(states_by_b):
                n = int(box_target_mask[b].sum().item())
                m = min(n, int(states.shape[0]))
                if m > 0:
                    pred_list.append(self.heads.decode_boxes(states[:m]))
                    tgt_list.append(box_targets[b, :m].to(states.device, states.dtype))
            if pred_list:
                pred = torch.cat(pred_list, dim=0)
                tgt = torch.cat(tgt_list, dim=0)
                box_l1 = F.l1_loss(pred, tgt)
                box_giou = generalized_iou_loss(pred, tgt)
                loss = loss + self.loss_weights["box_l1"] * box_l1 + self.loss_weights["box_giou"] * box_giou
                logs["box_l1"] = box_l1.detach()
                logs["box_giou"] = box_giou.detach()

        if pose_targets is not None and pose_target_mask is not None:
            states_by_b = self._token_states(input_ids, hidden, self.token_ids["pose"], labels=labels)
            xy_losses, vis_losses = [], []
            for b, states in enumerate(states_by_b):
                n = int(pose_target_mask[b].sum().item())
                m = min(n, int(states.shape[0]))
                if m <= 0:
                    continue
                xy, vis_logits = self.heads.decode_pose(states[:m])
                tgt = pose_targets[b, :m].to(states.device, states.dtype)
                tgt_xy = tgt[..., :2]
                tgt_v = tgt[..., 2].long().clamp(0, 2)
                visible = tgt_v.gt(0).unsqueeze(-1)
                if visible.any():
                    xy_losses.append(F.l1_loss(xy[visible.expand_as(xy)], tgt_xy[visible.expand_as(tgt_xy)]))
                vis_losses.append(F.cross_entropy(vis_logits.reshape(-1, 3), tgt_v.reshape(-1)))
            if xy_losses:
                pose_xy = torch.stack(xy_losses).mean()
                loss = loss + self.loss_weights["pose_xy"] * pose_xy
                logs["pose_xy"] = pose_xy.detach()
            if vis_losses:
                pose_vis = torch.stack(vis_losses).mean()
                loss = loss + self.loss_weights["pose_vis"] * pose_vis
                logs["pose_vis"] = pose_vis.detach()

        logs["loss"] = loss
        return logs

    @torch.no_grad()
    def decode_generated_heads(self, input_ids: torch.Tensor, hidden: torch.Tensor, start_pos: int) -> Dict[str, Any]:
        box_states = self._token_states(input_ids, hidden, self.token_ids["box"], labels=None, start_pos=start_pos)[0]
        pose_states = self._token_states(input_ids, hidden, self.token_ids["pose"], labels=None, start_pos=start_pos)[0]
        seg_states = self._token_states(input_ids, hidden, self.token_ids["seg"], labels=None, start_pos=start_pos)[0]
        head_dtype = next(self.heads.parameters()).dtype
        box_states = box_states.to(dtype=head_dtype)
        pose_states = pose_states.to(dtype=head_dtype)
        seg_states = seg_states.to(dtype=head_dtype)
        boxes = self.heads.decode_boxes(box_states).float().cpu().tolist() if box_states.numel() else []
        poses: List[List[List[float]]] = []
        if pose_states.numel():
            xy, vis_logits = self.heads.decode_pose(pose_states)
            vis = vis_logits.argmax(dim=-1).float().unsqueeze(-1)
            pose = torch.cat([xy.float(), vis], dim=-1)
            poses = pose.cpu().tolist()
        seg_embeddings = self.heads.seg_projector(seg_states).float().cpu().tolist() if seg_states.numel() else []
        return {"boxes_norm": boxes, "poses_norm": poses, "seg_embeddings": seg_embeddings}

    def save_heads(self, output_dir: str) -> None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        torch.save({"heads": self.heads.state_dict(), "token_ids": self.token_ids, "loss_weights": self.loss_weights}, out / "token_heads.pt")
        if hasattr(self.base_model, "save_pretrained"):
            self.base_model.save_pretrained(out / "base_or_adapter")

    def load_heads(self, path: str, map_location: str = "cpu") -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.heads.load_state_dict(ckpt["heads"], strict=True)
