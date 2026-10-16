from __future__ import annotations

import json
from pathlib import Path

import torch

from qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone import FrozenQwenDINOv3Backbone


def main() -> None:
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FrozenQwenDINOv3Backbone(
        qwen_model_name_or_path="/share2/program/weiminghong/model/Qwen3.5-4B",
        qwen_adapter_path="/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/legacy_sft/checkpoint-8000",
        qwen_torch_dtype="bfloat16",
        qwen_merge_lora=True,
        model_name_or_path="/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m",
        out_indices=(-4, -3, -2, -1),
        out_channels=64,
        num_outs=4,
        patch_size=16,
        freeze_backbone=True,
        fusion_mode="feature_add",
        bridge_hidden_channels=16,
        freeze_qwen=True,
    ).to(device)
    model.eval()
    qwen_total = sum(p.numel() for p in model.qwen_vision.parameters())
    qwen_trainable = sum(p.numel() for p in model.qwen_vision.parameters() if p.requires_grad)
    dino_trainable = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    x = torch.randn(1, 3, 224, 224, device=device)
    with torch.no_grad():
        feats = model(x)
    summary = {
        "device": device,
        "fusion_mode": model.fusion_mode,
        "qwen_total_params": qwen_total,
        "qwen_trainable_params": qwen_trainable,
        "dino_trainable_params": dino_trainable,
        "num_features": len(feats),
        "feature_shapes": [list(t.shape) for t in feats],
        "feature_dtypes": [str(t.dtype) for t in feats],
    }
    out = Path("/share2/program/weiminghong/humanslot_outputs/stage2_smoke/qwen_ie_dinov3_backbone_smoke.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
