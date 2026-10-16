from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone import FrozenQwenDINOv3Backbone


DEFAULT_QWEN_MODEL = "/share2/program/weiminghong/model/Qwen3.5-4B"
DEFAULT_QWEN_ADAPTER = "/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/legacy_sft/checkpoint-8000"
DEFAULT_DINO = "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_JSONL = "/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/train.jsonl"
DEFAULT_OUT = "/share2/program/weiminghong/humanslot_outputs/stage2_smoke/humanv2_stage2_train_smoke.json"
MMDET_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
MMDET_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)


class SimpleStage2Heads(nn.Module):
    def __init__(self, in_channels: int = 64, hidden_channels: int = 128, mask_size: int = 14) -> None:
        super().__init__()
        self.mask_size = int(mask_size)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.box_head = nn.Linear(hidden_channels, 4)
        self.pose_head = nn.Linear(hidden_channels, 17 * 2)
        self.gender_head = nn.Linear(hidden_channels, 3)
        self.age_head = nn.Linear(hidden_channels, 1)
        self.mask_head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = self.pool(feat).flatten(1)
        hidden = self.mlp(pooled)
        return {
            "box_xyxy": torch.sigmoid(self.box_head(hidden)),
            "pose_xy": torch.sigmoid(self.pose_head(hidden)).view(-1, 17, 2),
            "gender_logits": self.gender_head(hidden),
            "age": self.age_head(hidden).squeeze(-1),
            "mask_logits": F.interpolate(
                self.mask_head(feat),
                size=(self.mask_size, self.mask_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="human_v2-only Stage2 train smoke with frozen Qwen IE and DINOv3.")
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-adapter", default=DEFAULT_QWEN_ADAPTER)
    parser.add_argument("--dino-model", default=DEFAULT_DINO)
    parser.add_argument("--output-json", default=DEFAULT_OUT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_first_record(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                return json.loads(line)
    raise ValueError(f"No records in {path}")


def _load_image_tensor(record: Dict, image_size: int) -> torch.Tensor:
    image_path = Path(record.get("image_abs") or record.get("image"))
    if not image_path.exists():
        raise FileNotFoundError(str(image_path))
    image = Image.open(image_path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    return ((arr - MMDET_MEAN) / MMDET_STD).unsqueeze(0)


def _make_targets(record: Dict, mask_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
    width = float(record["width"])
    height = float(record["height"])
    box = record["box"][0]["bbox_xyxy"]
    box_target = torch.tensor(
        [box[0] / width, box[1] / height, box[2] / width, box[3] / height],
        dtype=torch.float32,
        device=device,
    ).clamp(0, 1).unsqueeze(0)

    pose_xy = torch.zeros(1, 17, 2, dtype=torch.float32, device=device)
    pose_visible = torch.zeros(1, 17, dtype=torch.float32, device=device)
    if record.get("pose"):
        for idx, point in enumerate(record["pose"][0].get("keypoints", [])[:17]):
            x, y, v = float(point[0]), float(point[1]), float(point[2])
            if v > 0:
                pose_xy[0, idx, 0] = x / width
                pose_xy[0, idx, 1] = y / height
                pose_visible[0, idx] = 1.0

    attr = record.get("attribute") or []
    gender = -1
    age = -1.0
    if attr:
        gender = int(attr[0].get("gender", -1))
        age = float(attr[0].get("age", -1))
    gender_target = torch.tensor([gender if gender in (0, 1, 2) else 2], dtype=torch.long, device=device)
    gender_weight = torch.tensor([1.0 if gender in (0, 1) else 0.0], dtype=torch.float32, device=device)
    age_target = torch.tensor([age / 100.0 if age >= 0 else 0.0], dtype=torch.float32, device=device)
    age_weight = torch.tensor([1.0 if age >= 0 else 0.0], dtype=torch.float32, device=device)

    mask = torch.zeros(1, mask_size, mask_size, dtype=torch.float32, device=device)
    x1, y1, x2, y2 = box_target[0]
    ix1 = int(torch.floor(x1 * mask_size).item())
    iy1 = int(torch.floor(y1 * mask_size).item())
    ix2 = max(ix1 + 1, int(torch.ceil(x2 * mask_size).item()))
    iy2 = max(iy1 + 1, int(torch.ceil(y2 * mask_size).item()))
    mask[:, max(0, iy1): min(mask_size, iy2), max(0, ix1): min(mask_size, ix2)] = 1.0
    return {
        "box_xyxy": box_target,
        "pose_xy": pose_xy,
        "pose_visible": pose_visible,
        "gender": gender_target,
        "gender_weight": gender_weight,
        "age": age_target,
        "age_weight": age_weight,
        "mask": mask,
    }



def _is_initialized_param(param: torch.nn.Parameter) -> bool:
    return not isinstance(param, torch.nn.parameter.UninitializedParameter)


def _initialized_trainable_params(module: nn.Module):
    return [p for p in module.parameters() if p.requires_grad and _is_initialized_param(p)]

def _compute_losses(pred: Dict[str, torch.Tensor], target: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
    bbox_loss = F.l1_loss(pred["box_xyxy"], target["box_xyxy"])
    pose_delta = (pred["pose_xy"] - target["pose_xy"]).abs() * target["pose_visible"].unsqueeze(-1)
    pose_loss = pose_delta.sum() / target["pose_visible"].sum().clamp_min(1.0)
    mask_loss = F.binary_cross_entropy_with_logits(pred["mask_logits"], target["mask"])
    gender_raw = F.cross_entropy(pred["gender_logits"], target["gender"], reduction="none")
    gender_loss = (gender_raw * target["gender_weight"]).sum() / target["gender_weight"].sum().clamp_min(1.0)
    age_raw = F.l1_loss(pred["age"], target["age"], reduction="none")
    age_loss = (age_raw * target["age_weight"]).sum() / target["age_weight"].sum().clamp_min(1.0)
    loss = bbox_loss + 0.5 * pose_loss + mask_loss + gender_loss + age_loss
    parts = {
        "bbox_loss": float(bbox_loss.detach().cpu()),
        "pose_loss": float(pose_loss.detach().cpu()),
        "mask_loss": float(mask_loss.detach().cpu()),
        "gender_loss": float(gender_loss.detach().cpu()),
        "age_loss": float(age_loss.detach().cpu()),
        "total_loss": float(loss.detach().cpu()),
    }
    return loss, parts


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    record = _load_first_record(args.jsonl)
    image = _load_image_tensor(record, args.image_size).to(device)
    target = _make_targets(record, mask_size=14, device=device)

    backbone = FrozenQwenDINOv3Backbone(
        qwen_model_name_or_path=args.qwen_model,
        qwen_adapter_path=args.qwen_adapter,
        qwen_torch_dtype="bfloat16",
        qwen_merge_lora=True,
        model_name_or_path=args.dino_model,
        out_indices=(-4, -3, -2, -1),
        out_channels=64,
        num_outs=4,
        patch_size=16,
        freeze_backbone=True,
        fusion_mode="feature_add",
        bridge_hidden_channels=16,
        freeze_qwen=True,
    ).to(device)
    heads = SimpleStage2Heads(in_channels=64).to(device)
    backbone.train()
    heads.train()

    with torch.no_grad():
        _ = backbone(image)
    trainable_params = _initialized_trainable_params(backbone) + _initialized_trainable_params(heads)
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    history = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        feat = backbone(image)[0]
        pred = heads(feat)
        loss, parts = _compute_losses(pred, target)
        loss.backward()
        optimizer.step()
        parts["step"] = step + 1
        history.append(parts)

    qwen_trainable_params = sum(p.numel() for p in backbone.qwen_vision.parameters() if p.requires_grad)
    dino_trainable_params = sum(p.numel() for p in backbone.backbone.parameters() if p.requires_grad)
    summary = {
        "device": str(device),
        "record_id": record.get("id"),
        "image": record.get("image_abs"),
        "fusion_mode": backbone.fusion_mode,
        "qwen_trainable_params": qwen_trainable_params,
        "dino_trainable_params": dino_trainable_params,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "history": history,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
