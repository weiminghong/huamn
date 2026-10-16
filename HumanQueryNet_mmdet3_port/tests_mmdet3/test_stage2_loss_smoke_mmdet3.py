from __future__ import annotations

import json

import torch
from mmengine.config import Config
from mmdet.registry import MODELS

import models  # noqa: F401
import core  # noqa: F401
from compat.stage2_humanv2_adapter import build_humanv2_stage2_batch

ANN = "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
IMG_ROOT = "/share2/program/weiminghong/coco/val2017"


def _smoke_cfg() -> Config:
    cfg = Config.fromfile("configs/human_query_net_all_task_r50_scratch.py")
    cfg.model.backbone.init_cfg = None
    cfg.model.bbox_head.with_smpl = False
    cfg.model.bbox_head.pop("smpl_cfg", None)
    cfg.model.bbox_head.num_query = 24
    cfg.model.bbox_head.transformer.encoder.num_layers = 1
    cfg.model.bbox_head.transformer.decoder.num_layers = 1
    cfg.model.bbox_head.transformer.kpt_refine_decoder.num_layers = 1
    cfg.model.bbox_head.transformer.hm_encoder.num_layers = 1
    cfg.model.bbox_head.dn_cfg.group_cfg.num_dn_queries = 8
    cfg.model.bbox_head.seg_cfg.train_cfg.num_points = 256
    cfg.model.train_cfg.uni_assigner.kpt_cost.weight = cfg.model.bbox_head.kp_cfg.loss_kpt.loss_weight
    cfg.model.train_cfg.uni_assigner.oks_cost.weight = cfg.model.bbox_head.kp_cfg.loss_oks.loss_weight
    cfg.model.train_cfg.uni_assigner.mask_cost.weight = cfg.model.bbox_head.seg_cfg.loss_mask.loss_weight
    cfg.model.train_cfg.uni_assigner.dice_cost.weight = cfg.model.bbox_head.seg_cfg.loss_dice.loss_weight
    return cfg


def _sum_losses(losses: dict) -> torch.Tensor:
    total = None
    for value in losses.values():
        if isinstance(value, torch.Tensor):
            item = value.mean()
        elif isinstance(value, (list, tuple)):
            tensors = [v.mean() for v in value if isinstance(v, torch.Tensor)]
            if not tensors:
                continue
            item = sum(tensors)
        else:
            continue
        total = item if total is None else total + item
    assert total is not None
    return total


def main() -> None:
    assert torch.cuda.is_available(), "loss smoke requires CUDA because legacy code still contains .cuda() paths"
    device = torch.device("cuda")
    torch.manual_seed(7)
    cfg = _smoke_cfg()
    model = MODELS.build(cfg.model).to(device)
    model.train()
    batch = build_humanv2_stage2_batch(
        ann_file=ANN,
        image_root=IMG_ROOT,
        image_size=128,
        max_instances=2,
        require_keypoints=True,
        device=device,
    )
    losses = model.forward_train(
        batch["img"],
        batch["img_metas"],
        batch["gt_bboxes"],
        batch["gt_labels"],
        batch["gt_attributes"],
        batch["gt_keypoints"],
        batch["gt_areas"],
        batch["gt_masks"],
        batch["gt_valids"],
        batch["gt_smpl_pose"],
        batch["gt_smpl_betas"],
    )
    total = _sum_losses(losses)
    assert torch.isfinite(total).item(), losses
    total.backward()
    summary = {
        "sample": batch["sample_info"],
        "num_loss_terms": len(losses),
        "total_loss": float(total.detach().cpu()),
        "loss_keys": sorted(losses.keys())[:20],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
