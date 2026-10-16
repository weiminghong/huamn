from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from mmengine.config import Config
from mmdet.registry import MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_PORT = REPO_ROOT / "HumanQueryNet_mmdet3_port"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HQNET_PORT) not in sys.path:
    sys.path.insert(0, str(HQNET_PORT))

import models  # noqa: F401,E402
import core  # noqa: F401,E402
from compat.stage2_humanv2_adapter import build_humanv2_stage2_batch  # noqa: E402
from qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone import FrozenQwenDINOv3Backbone  # noqa: F401,E402

DEFAULT_CONFIG = REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage2_qwen_ie_dinov3_hqnet.py"
DEFAULT_QWEN = "/share2/program/weiminghong/model/Qwen3.5-4B"
DEFAULT_ADAPTER = "/share2/program/weiminghong/humanslot_outputs/stage12_smoke_20260709/stage1_qwen_tiny224_len768_smoke"
DEFAULT_DINO = "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_ANN = "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
DEFAULT_IMG_ROOT = "/share2/program/weiminghong/coco/val2017"
DEFAULT_OUT = "/share2/program/weiminghong/humanslot_outputs/stage12_smoke_20260709/stage2_qwen_dino_hqnet_loss_smoke.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage2 Qwen+DINOv3+HumanQueryNet mmdet3 loss smoke.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN)
    parser.add_argument("--qwen-adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--dino-model", default=DEFAULT_DINO)
    parser.add_argument("--ann-file", default=DEFAULT_ANN)
    parser.add_argument("--image-root", default=DEFAULT_IMG_ROOT)
    parser.add_argument("--output-json", default=DEFAULT_OUT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-instances", type=int, default=2)
    parser.add_argument("--num-query", type=int, default=16)
    parser.add_argument("--fusion-mode", default="feature_add")
    return parser.parse_args()


def make_smoke_cfg(args: argparse.Namespace) -> Config:
    cfg = Config.fromfile(args.config)
    cfg.model.backbone.qwen_model_name_or_path = args.qwen_model
    cfg.model.backbone.qwen_adapter_path = args.qwen_adapter
    cfg.model.backbone.model_name_or_path = args.dino_model
    cfg.model.backbone.qwen_merge_lora = True
    cfg.model.backbone.freeze_qwen = True
    cfg.model.backbone.freeze_backbone = True
    cfg.model.backbone.fusion_mode = args.fusion_mode
    cfg.model.backbone.out_channels = 64
    cfg.model.backbone.bridge_hidden_channels = 16
    cfg.model.backbone.num_outs = 5
    cfg.model.backbone.target_strides = [4, 8, 16, 32, 64]
    cfg.model.neck.in_channels = [64, 64, 64, 64, 64]
    cfg.model.neck.num_outs = 5

    cfg.model.bbox_head.with_smpl = False
    cfg.model.bbox_head.pop("smpl_cfg", None)
    cfg.model.bbox_head.num_query = int(args.num_query)
    cfg.model.bbox_head.transformer.encoder.num_layers = 1
    cfg.model.bbox_head.transformer.decoder.num_layers = 1
    cfg.model.bbox_head.transformer.kpt_refine_decoder.num_layers = 1
    cfg.model.bbox_head.transformer.hm_encoder.num_layers = 1
    cfg.model.bbox_head.dn_cfg.group_cfg.num_dn_queries = max(4, int(args.num_query) // 2)
    cfg.model.bbox_head.seg_cfg.train_cfg.num_points = 256
    cfg.model.train_cfg.uni_assigner.kpt_cost.weight = cfg.model.bbox_head.kp_cfg.loss_kpt.loss_weight
    cfg.model.train_cfg.uni_assigner.oks_cost.weight = cfg.model.bbox_head.kp_cfg.loss_oks.loss_weight
    cfg.model.train_cfg.uni_assigner.mask_cost.weight = cfg.model.bbox_head.seg_cfg.loss_mask.loss_weight
    cfg.model.train_cfg.uni_assigner.dice_cost.weight = cfg.model.bbox_head.seg_cfg.loss_dice.loss_weight
    return cfg


def initialized_trainable_params(module: torch.nn.Module):
    return [
        p for p in module.parameters()
        if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)
    ]


def sum_losses(losses: dict) -> torch.Tensor:
    total = None
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            item = value.mean()
        elif isinstance(value, (list, tuple)):
            terms = [v.mean() for v in value if isinstance(v, torch.Tensor)]
            if not terms:
                continue
            item = sum(terms)
        else:
            continue
        total = item if total is None else total + item
    if total is None:
        raise RuntimeError("No tensor loss terms returned")
    return total


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this smoke because legacy code still has CUDA-only paths")
    device = torch.device("cuda")
    torch.manual_seed(11)
    cfg = make_smoke_cfg(args)
    model = MODELS.build(cfg.model).to(device)
    model.train()
    batch = build_humanv2_stage2_batch(
        ann_file=args.ann_file,
        image_root=args.image_root,
        image_size=args.image_size,
        max_instances=args.max_instances,
        require_keypoints=True,
        device=device,
    )
    trainable_params = initialized_trainable_params(model)
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    losses = model.forward_train(
        batch["img"], batch["img_metas"], batch["gt_bboxes"], batch["gt_labels"],
        batch["gt_attributes"], batch["gt_keypoints"], batch["gt_areas"], batch["gt_masks"],
        batch["gt_valids"], batch["gt_smpl_pose"], batch["gt_smpl_betas"],
    )
    total = sum_losses(losses)
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite loss: {total}")
    total.backward()
    optimizer.step()
    qwen_trainable = sum(p.numel() for p in model.backbone.qwen_vision.parameters() if p.requires_grad)
    dino_trainable = sum(p.numel() for p in model.backbone.backbone.parameters() if p.requires_grad)
    summary = {
        "sample": batch["sample_info"],
        "image_size": args.image_size,
        "fusion_mode": cfg.model.backbone.fusion_mode,
        "qwen_adapter": args.qwen_adapter,
        "qwen_trainable_params": qwen_trainable,
        "dino_trainable_params": dino_trainable,
        "initialized_trainable_params": sum(p.numel() for p in trainable_params),
        "num_loss_terms": len(losses),
        "total_loss": float(total.detach().cpu()),
        "loss_keys": sorted(losses.keys()),
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
