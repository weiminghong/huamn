from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from mmengine.config import Config
from mmdet.registry import MODELS
from mmdet.structures.mask import BitmapMasks

REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_PORT = REPO_ROOT / "HumanQueryNet_mmdet3_port"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HQNET_PORT) not in sys.path:
    sys.path.insert(0, str(HQNET_PORT))

import models  # noqa: F401,E402
import core  # noqa: F401,E402
from compat.stage2_humanv2_adapter import (  # noqa: E402
    _age_value,
    _annotation_has_valid_box,
    _decode_mask,
    _gender_value,
    _resize_image_tensor,
    _resize_mask,
    _scale_keypoints,
    _valid_vector,
)
from qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone import FrozenQwenDINOv3Backbone  # noqa: F401,E402

DEFAULT_CONFIG = REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage2_qwen_ie_dinov3_hqnet.py"
DEFAULT_BASELINE_CONFIG = REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage2_dinov3_hqnet_baseline.py"
DEFAULT_QWEN = "/share2/program/weiminghong/model/Qwen3.5-4B"
DEFAULT_ADAPTER = ""
DEFAULT_DINO = "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m"
DEFAULT_TRAIN_ANN = "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json"
DEFAULT_TRAIN_ROOT = "/share2/program/weiminghong/coco/train2017"
DEFAULT_WORK_DIR = "/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260709"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native full-dataset Stage2 training for Qwen+DINOv3+HumanQueryNet.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone-mode", choices=["qwen_fusion", "dino_only"], default="qwen_fusion")
    parser.add_argument("--train-ann-file", default=DEFAULT_TRAIN_ANN)
    parser.add_argument("--train-image-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--qwen-model", default=DEFAULT_QWEN)
    parser.add_argument("--qwen-adapter", default=DEFAULT_ADAPTER, help="Required Stage1-completed Qwen adapter directory. Do not use smoke or legacy checkpoints for official Stage2.")
    parser.add_argument("--dino-model", default=DEFAULT_DINO)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument("--profile", choices=["full", "lite"], default="full", help="full keeps the HumanQueryNet-scale head; lite keeps the old 224/100-query diagnostic overrides.")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-query", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Accumulate gradients over this many physical batches before optimizer.step().")
    parser.add_argument("--max-epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0, help="0 means no step cap")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for bridge/neck/task heads.")
    parser.add_argument("--dino-lr", type=float, default=1e-5, help="Learning rate for trainable DINOv3 backbone parameters. Defaults to 0.1x head LR.")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=0.1)
    parser.add_argument("--bbox-loss-weight", type=float, default=-1.0, help="Override bbox_head.loss_bbox.loss_weight when >=0.")
    parser.add_argument("--iou-loss-weight", type=float, default=-1.0, help="Override bbox_head.loss_iou.loss_weight when >=0.")
    parser.add_argument("--cls-loss-weight", type=float, default=-1.0, help="Override bbox_head.loss_cls.loss_weight when >=0.")
    parser.add_argument("--kpt-cls-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_kpt_cls.loss_weight when >=0.")
    parser.add_argument("--hm-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_hm.loss_weight when >=0. Useful for fullcfg warmup stability.")
    parser.add_argument("--kpt-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_kpt.loss_weight when >=0.")
    parser.add_argument("--kpt-rpn-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_kpt_rpn.loss_weight when >=0.")
    parser.add_argument("--oks-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_oks.loss_weight when >=0.")
    parser.add_argument("--kpt-refine-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_kpt_refine.loss_weight when >=0.")
    parser.add_argument("--oks-refine-loss-weight", type=float, default=-1.0, help="Override kp_cfg.loss_oks_refine.loss_weight when >=0.")
    parser.add_argument("--mask-loss-weight", type=float, default=-1.0, help="Override seg_cfg.loss_mask.loss_weight when >=0.")
    parser.add_argument("--dice-loss-weight", type=float, default=-1.0, help="Override seg_cfg.loss_dice.loss_weight when >=0.")
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fusion-mode", default="feature_add")
    parser.add_argument("--freeze-dino", action="store_true", help="Freeze DINOv3. Default trains DINOv3 for Stage2.")
    parser.add_argument("--max-instances", type=int, default=0, help="0 keeps all instances")
    parser.add_argument("--limit-images", type=int, default=0, help="0 uses the full indexed train set")
    parser.add_argument("--skip-sample-errors", action="store_true")
    parser.add_argument("--shared-init-state", default="", help="Optional shared DINO/neck/head initialization used for paired ablations.")
    parser.add_argument("--write-shared-init", action="store_true", help="Write --shared-init-state after lazy layers initialize when the file does not exist.")
    return parser.parse_args()


def make_cfg(args: argparse.Namespace) -> Config:
    backbone_mode = str(getattr(args, "backbone_mode", "qwen_fusion"))
    config_path = str(DEFAULT_BASELINE_CONFIG) if backbone_mode == "dino_only" else args.config
    cfg = Config.fromfile(config_path)
    cfg.model.backbone.model_name_or_path = args.dino_model
    cfg.model.backbone.freeze_backbone = bool(args.freeze_dino)
    if backbone_mode == "qwen_fusion":
        cfg.model.backbone.qwen_model_name_or_path = args.qwen_model
        cfg.model.backbone.qwen_adapter_path = args.qwen_adapter
        cfg.model.backbone.qwen_merge_lora = True
        cfg.model.backbone.freeze_qwen = True
        cfg.model.backbone.fusion_mode = args.fusion_mode

    profile = str(getattr(args, "profile", "full"))
    if profile == "lite":
        cfg.model.backbone.out_channels = 64
        if backbone_mode == "qwen_fusion":
            cfg.model.backbone.bridge_hidden_channels = 16
        cfg.model.backbone.num_outs = 5
        cfg.model.backbone.target_strides = [4, 8, 16, 32, 64]
        cfg.model.neck.in_channels = [64, 64, 64, 64, 64]
        cfg.model.neck.num_outs = 5
        cfg.model.bbox_head.transformer.encoder.num_layers = 1
        cfg.model.bbox_head.transformer.decoder.num_layers = 1
        cfg.model.bbox_head.transformer.kpt_refine_decoder.num_layers = 1
        cfg.model.bbox_head.transformer.hm_encoder.num_layers = 1
        cfg.model.bbox_head.seg_cfg.train_cfg.num_points = 512

    cfg.model.bbox_head.with_smpl = False
    cfg.model.bbox_head.pop("smpl_cfg", None)
    cfg.model.bbox_head.num_query = int(args.num_query)
    cfg.model.bbox_head.dn_cfg.group_cfg.num_dn_queries = max(8, int(args.num_query) // 2)
    if float(getattr(args, "bbox_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.loss_bbox.loss_weight = float(args.bbox_loss_weight)
    if float(getattr(args, "iou_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.loss_iou.loss_weight = float(args.iou_loss_weight)
    if float(getattr(args, "cls_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.loss_cls.loss_weight = float(args.cls_loss_weight)
    if float(getattr(args, "kpt_cls_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_kpt_cls.loss_weight = float(args.kpt_cls_loss_weight)
    if float(getattr(args, "hm_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_hm.loss_weight = float(args.hm_loss_weight)
    if float(getattr(args, "kpt_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_kpt.loss_weight = float(args.kpt_loss_weight)
    if float(getattr(args, "kpt_rpn_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_kpt_rpn.loss_weight = float(args.kpt_rpn_loss_weight)
    if float(getattr(args, "oks_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_oks.loss_weight = float(args.oks_loss_weight)
    if float(getattr(args, "kpt_refine_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_kpt_refine.loss_weight = float(args.kpt_refine_loss_weight)
    if float(getattr(args, "oks_refine_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.kp_cfg.loss_oks_refine.loss_weight = float(args.oks_refine_loss_weight)
    if float(getattr(args, "mask_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.seg_cfg.loss_mask.loss_weight = float(args.mask_loss_weight)
    if float(getattr(args, "dice_loss_weight", -1.0)) >= 0:
        cfg.model.bbox_head.seg_cfg.loss_dice.loss_weight = float(args.dice_loss_weight)
    cfg.model.train_cfg.uni_assigner.reg_cost.weight = cfg.model.bbox_head.loss_bbox.loss_weight
    cfg.model.train_cfg.uni_assigner.iou_cost.weight = cfg.model.bbox_head.loss_iou.loss_weight
    cfg.model.train_cfg.uni_assigner.cls_cost.weight = cfg.model.bbox_head.loss_cls.loss_weight
    cfg.model.train_cfg.uni_assigner.kpt_cls_cost.weight = cfg.model.bbox_head.kp_cfg.loss_kpt_cls.loss_weight
    cfg.model.train_cfg.uni_assigner.kpt_cost.weight = cfg.model.bbox_head.kp_cfg.loss_kpt.loss_weight
    cfg.model.train_cfg.uni_assigner.oks_cost.weight = cfg.model.bbox_head.kp_cfg.loss_oks.loss_weight
    cfg.model.train_cfg.uni_assigner.mask_cost.weight = cfg.model.bbox_head.seg_cfg.loss_mask.loss_weight
    cfg.model.train_cfg.uni_assigner.dice_cost.weight = cfg.model.bbox_head.seg_cfg.loss_dice.loss_weight
    return cfg


def load_index(ann_file: str, image_root: str, limit_images: int = 0) -> List[Dict[str, Any]]:
    print(json.dumps({"event": "json_load_start", "ann_file": ann_file}, ensure_ascii=False), flush=True)
    data = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    print(json.dumps({"event": "json_loaded", "images": len(data.get("images", [])), "annotations": len(data.get("annotations", []))}, ensure_ascii=False), flush=True)
    anns_by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        anns_by_image.setdefault(int(ann["image_id"]), []).append(ann)

    root = Path(image_root)
    samples: List[Dict[str, Any]] = []
    missing_images = 0
    for img in data.get("images", []):
        img_id = int(img.get("id", img.get("img_id")))
        width, height = int(img["width"]), int(img["height"])
        anns = [
            a for a in anns_by_image.get(img_id, [])
            if not a.get("iscrowd", False) and _annotation_has_valid_box(a, width, height)
        ]
        if not anns:
            continue
        file_name = img.get("file_name") or img.get("filename")
        if not file_name:
            continue
        img_path = root / file_name
        samples.append({"img": img, "anns": anns, "path": str(img_path)})
        if limit_images and len(samples) >= limit_images:
            break
    if not samples:
        raise RuntimeError(f"No usable samples indexed from {ann_file}")
    print(json.dumps({"event": "index_built", "samples": len(samples), "missing_images": missing_images}, ensure_ascii=False), flush=True)
    return samples


def build_batch(sample: Dict[str, Any], image_size: int, max_instances: int, device: torch.device) -> Dict[str, Any]:
    img_info = sample["img"]
    anns = list(sample["anns"])
    if max_instances > 0:
        anns = anns[:max_instances]
    img_path = Path(sample["path"])
    image_tensor, (ori_h, ori_w) = _resize_image_tensor(img_path, image_size)
    sx, sy = image_size / float(ori_w), image_size / float(ori_h)

    bboxes: List[List[float]] = []
    labels: List[int] = []
    attributes: List[List[int]] = []
    keypoints: List[List[float]] = []
    areas: List[float] = []
    valids: List[List[int]] = []
    smpl_pose: List[List[float]] = []
    smpl_betas: List[List[float]] = []
    masks: List[np.ndarray] = []

    for ann in anns:
        x, y, w, h = map(float, ann["bbox"])
        x1 = max(0.0, min(float(image_size), x * sx))
        y1 = max(0.0, min(float(image_size), y * sy))
        x2 = max(0.0, min(float(image_size), (x + w) * sx))
        y2 = max(0.0, min(float(image_size), (y + h) * sy))
        if x2 <= x1 or y2 <= y1:
            continue
        valid = _valid_vector(ann)
        bboxes.append([x1, y1, x2, y2])
        labels.append(0)
        attributes.append([
            _gender_value(ann.get("gender")) if valid[6] else -1,
            _age_value(ann.get("age")) if valid[7] else -1,
        ])
        keypoints.append(_scale_keypoints(ann, sx, sy, image_size) if valid[2] else [0.0] * 51)
        areas.append(max(1.0, float(ann.get("area", w * h)) * sx * sy))
        valids.append(valid)
        smpl = ann.get("smpl") if valid[5] else None
        smpl_pose.append(list((smpl or {}).get("pose", [0.0] * 72))[:72])
        smpl_betas.append(list((smpl or {}).get("betas", [0.0] * 10))[:10])
        mask = _decode_mask(ann.get("segmentation") if valid[4] else None, ori_h, ori_w)
        masks.append(_resize_mask(mask, image_size))

    if not bboxes:
        raise RuntimeError("sample has no valid boxes after resize")

    img = image_tensor.unsqueeze(0).to(device)
    return {
        "img": img,
        "img_metas": [{
            "filename": str(img_path),
            "ori_filename": img_path.name,
            "img_shape": (image_size, image_size, 3),
            "ori_shape": (ori_h, ori_w, 3),
            "pad_shape": (image_size, image_size, 3),
            "batch_input_shape": (image_size, image_size),
            "scale_factor": np.array([sx, sy, sx, sy], dtype=np.float32),
            "flip": False,
            "flip_direction": None,
            "img_norm_cfg": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True},
        }],
        "gt_bboxes": [torch.tensor(bboxes, dtype=torch.float32, device=device)],
        "gt_labels": [torch.tensor(labels, dtype=torch.long, device=device)],
        "gt_attributes": [torch.tensor(attributes, dtype=torch.long, device=device)],
        "gt_keypoints": [torch.tensor(keypoints, dtype=torch.float32, device=device)],
        "gt_areas": [torch.tensor(areas, dtype=torch.float32, device=device)],
        "gt_masks": [BitmapMasks(np.stack(masks, axis=0), image_size, image_size)],
        "gt_valids": [torch.tensor(valids, dtype=torch.long, device=device)],
        "gt_smpl_pose": [torch.tensor(smpl_pose, dtype=torch.float32, device=device)],
        "gt_smpl_betas": [torch.tensor(smpl_betas, dtype=torch.float32, device=device)],
        "sample_info": {"img_id": int(img_info.get("id", img_info.get("img_id"))), "img_path": str(img_path), "num_instances": len(bboxes)},
    }


def collate_batches(items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not items:
        raise RuntimeError("empty batch")
    return {
        "img": torch.cat([item["img"] for item in items], dim=0),
        "img_metas": [meta for item in items for meta in item["img_metas"]],
        "gt_bboxes": [x for item in items for x in item["gt_bboxes"]],
        "gt_labels": [x for item in items for x in item["gt_labels"]],
        "gt_attributes": [x for item in items for x in item["gt_attributes"]],
        "gt_keypoints": [x for item in items for x in item["gt_keypoints"]],
        "gt_areas": [x for item in items for x in item["gt_areas"]],
        "gt_masks": [x for item in items for x in item["gt_masks"]],
        "gt_valids": [x for item in items for x in item["gt_valids"]],
        "gt_smpl_pose": [x for item in items for x in item["gt_smpl_pose"]],
        "gt_smpl_betas": [x for item in items for x in item["gt_smpl_betas"]],
        "sample_info": [item["sample_info"] for item in items],
    }


def initialized_trainable_params(module: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in module.parameters() if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)]


def build_optimizer_param_groups(model: torch.nn.Module, args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    dino_params: List[torch.nn.Parameter] = []
    head_params: List[torch.nn.Parameter] = []
    qwen_trainable_names: List[str] = []
    dino_names: List[str] = []
    head_names: List[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad or isinstance(param, torch.nn.parameter.UninitializedParameter):
            continue
        if name.startswith("backbone.qwen_vision."):
            qwen_trainable_names.append(name)
            continue
        if name.startswith("backbone.backbone."):
            dino_params.append(param)
            dino_names.append(name)
        else:
            head_params.append(param)
            head_names.append(name)
    if qwen_trainable_names:
        raise RuntimeError(f"Qwen vision/text parameters must be frozen in Stage2, but trainable params were found: {qwen_trainable_names[:5]}")

    groups: List[Dict[str, Any]] = []
    if head_params:
        groups.append({"params": head_params, "lr": args.lr, "weight_decay": args.weight_decay, "group_name": "bridge_neck_heads"})
    if dino_params:
        groups.append({"params": dino_params, "lr": args.dino_lr, "weight_decay": args.weight_decay, "group_name": "dinov3_backbone"})
    stats = {
        "optimizer_groups": [
            {"name": "bridge_neck_heads", "lr": args.lr, "weight_decay": args.weight_decay, "num_tensors": len(head_params), "num_params": sum(p.numel() for p in head_params), "example_names": head_names[:5]},
            {"name": "dinov3_backbone", "lr": args.dino_lr, "weight_decay": args.weight_decay, "num_tensors": len(dino_params), "num_params": sum(p.numel() for p in dino_params), "example_names": dino_names[:5]},
        ]
    }
    if not groups:
        raise RuntimeError("no trainable parameters for Stage2 optimizer")
    return groups, stats


def trainable_state_dict(module: torch.nn.Module) -> Dict[str, torch.Tensor]:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    keep = {name for name, p in params.items() if p.requires_grad and not isinstance(p, torch.nn.parameter.UninitializedParameter)}
    state = module.state_dict()
    out = {k: v.detach().cpu() for k, v in state.items() if k in keep or k in buffers and k.endswith("num_batches_tracked")}
    return out


def sum_losses(losses: Dict[str, Any]) -> torch.Tensor:
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
        raise RuntimeError("no tensor loss terms")
    return total


def clip_grad_norm_fp64(parameters: Sequence[torch.nn.Parameter], max_norm: float) -> Tuple[float, bool]:
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return 0.0, True
    finite = all(torch.isfinite(g).all().item() for g in grads)
    if not finite:
        return float("inf"), False
    total_sq = torch.zeros((), dtype=torch.float64, device=grads[0].device)
    for grad in grads:
        norm = torch.linalg.vector_norm(grad.double())
        total_sq = total_sq + norm * norm
    total_norm = torch.sqrt(total_sq)
    if max_norm and max_norm > 0:
        clip_coef = float(max_norm) / (float(total_norm.detach().cpu()) + 1e-12)
        if clip_coef < 1.0:
            for grad in grads:
                grad.mul_(clip_coef)
    return float(total_norm.detach().cpu()), True




class ReopenAppendWriter:
    def __init__(self, path: Path, encoding: str = "utf-8") -> None:
        self.path = Path(path)
        self.encoding = encoding

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def write(self, text: str) -> None:
        with self.path.open("a", encoding=self.encoding) as f:
            f.write(text)

    def flush(self) -> None:
        return None

def save_checkpoint(work_dir: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, epoch: int, args: argparse.Namespace, metrics: Dict[str, Any]) -> None:
    ckpt = {
        "step": step,
        "epoch": epoch,
        "args": vars(args),
        "metrics": metrics,
        "model_trainable": trainable_state_dict(model),
        "optimizer": optimizer.state_dict(),
    }
    path = work_dir / f"checkpoint_step_{step:07d}.pt"
    torch.save(ckpt, path)
    latest = work_dir / "latest.pt"
    torch.save(ckpt, latest)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.backbone_mode == "qwen_fusion" and not args.qwen_adapter:
        raise RuntimeError("--qwen-adapter is required for Stage2. Pass the completed Stage1 adapter path, e.g. .../train/final/base_or_adapter or an equivalent finalized adapter directory.")
    if args.backbone_mode == "qwen_fusion":
        adapter_path = Path(args.qwen_adapter)
        if not adapter_path.exists():
            raise RuntimeError(f"Stage1 adapter path does not exist: {adapter_path}")
        if "stage12_smoke" in str(adapter_path) or "legacy_sft" in str(adapter_path):
            raise RuntimeError(f"Refusing non-official Stage1 adapter for Stage2: {adapter_path}")

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    log_path = work_dir / "train_metrics.jsonl"
    manifest_path = work_dir / "manifest.json"

    samples = load_index(args.train_ann_file, args.train_image_root, args.limit_images)
    cfg = make_cfg(args)
    device = torch.device("cuda")
    model = MODELS.build(cfg.model).to(device)
    model.train()
    torch.cuda.reset_peak_memory_stats(device)

    max_instances = int(args.max_instances)
    warm_items = [build_batch(samples[i], args.image_size, max_instances, device) for i in range(min(args.batch_size, len(samples)))]
    first_batch = collate_batches(warm_items)
    with torch.no_grad():
        _ = model.extract_feat(first_batch["img"])

    shared_init_info: Dict[str, Any] = {"path": args.shared_init_state, "loaded": False, "written": False}
    if args.shared_init_state:
        shared_path = Path(args.shared_init_state)
        if shared_path.exists():
            saved = torch.load(shared_path, map_location="cpu")
            saved_state = saved.get("model", saved) if isinstance(saved, dict) else saved
            current_state = model.state_dict()
            compatible = {
                key: value for key, value in saved_state.items()
                if key in current_state and tuple(value.shape) == tuple(current_state[key].shape)
            }
            incompatible = model.load_state_dict(compatible, strict=False)
            shared_init_info.update({
                "loaded": True,
                "matched_keys": len(compatible),
                "missing_keys": len(incompatible.missing_keys),
                "unexpected_keys": len(incompatible.unexpected_keys),
            })
        elif args.write_shared_init:
            shared_path.parent.mkdir(parents=True, exist_ok=True)
            shared_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save({"model": shared_state, "backbone_mode": args.backbone_mode, "seed": args.seed}, shared_path)
            shared_init_info.update({"written": True, "matched_keys": len(shared_state)})
        else:
            raise FileNotFoundError(f"Shared initialization does not exist: {shared_path}")

    trainable_params = initialized_trainable_params(model)
    optimizer_groups, optimizer_stats = build_optimizer_param_groups(model, args)
    optimizer = torch.optim.AdamW(optimizer_groups)

    qwen_vision = getattr(model.backbone, "qwen_vision", None)
    manifest = {
        "train_ann_file": args.train_ann_file,
        "train_image_root": args.train_image_root,
        "num_samples": len(samples),
        "profile": args.profile,
        "image_size": args.image_size,
        "num_query": args.num_query,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * max(1, int(args.grad_accum_steps)),
        "loss_weights": {
            "bbox": args.bbox_loss_weight,
            "iou": args.iou_loss_weight,
            "cls": args.cls_loss_weight,
            "kpt_cls": args.kpt_cls_loss_weight,
            "hm": args.hm_loss_weight,
            "kpt": args.kpt_loss_weight,
            "kpt_rpn": args.kpt_rpn_loss_weight,
            "oks": args.oks_loss_weight,
            "kpt_refine": args.kpt_refine_loss_weight,
            "oks_refine": args.oks_refine_loss_weight,
            "mask": args.mask_loss_weight,
            "dice": args.dice_loss_weight,
        },
        "max_instances": args.max_instances,
        "qwen_model": args.qwen_model,
        "qwen_adapter": args.qwen_adapter,
        "backbone_mode": args.backbone_mode,
        "dino_model": args.dino_model,
        "fusion_mode": args.fusion_mode,
        "freeze_dino": args.freeze_dino,
        "head_lr": args.lr,
        "dino_lr": args.dino_lr,
        "shared_init": shared_init_info,
        "trainable_params": sum(p.numel() for p in trainable_params),
        "qwen_trainable_params": sum(p.numel() for p in qwen_vision.parameters() if p.requires_grad) if qwen_vision is not None else 0,
        "dino_trainable_params": sum(p.numel() for p in model.backbone.backbone.parameters() if p.requires_grad),
        **optimizer_stats,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"event": "train_start", **manifest}, ensure_ascii=False), flush=True)

    global_step = 0
    micro_step = 0
    skipped = 0
    accum_steps = max(1, int(args.grad_accum_steps))
    start_time = time.time()
    optimizer.zero_grad(set_to_none=True)
    with ReopenAppendWriter(log_path, encoding="utf-8") as log_f:
        for epoch in range(1, args.max_epochs + 1):
            order = list(range(len(samples)))
            random.Random(args.seed + epoch).shuffle(order)
            for batch_start in range(0, len(order), max(1, args.batch_size)):
                if args.max_steps and global_step >= args.max_steps:
                    break
                batch_indices = order[batch_start:batch_start + max(1, args.batch_size)]
                try:
                    items = [build_batch(samples[idx], args.image_size, max_instances, device) for idx in batch_indices]
                    batch = collate_batches(items)
                    losses = model.forward_train(
                        batch["img"], batch["img_metas"], batch["gt_bboxes"], batch["gt_labels"],
                        batch["gt_attributes"], batch["gt_keypoints"], batch["gt_areas"], batch["gt_masks"],
                        batch["gt_valids"], batch["gt_smpl_pose"], batch["gt_smpl_betas"],
                    )
                    total = sum_losses(losses)
                    if not torch.isfinite(total):
                        raise RuntimeError(f"non-finite loss {float(total.detach().cpu())}")
                    (total / accum_steps).backward()
                    micro_step += 1
                    is_last_micro = batch_start + max(1, args.batch_size) >= len(order)
                    did_optimizer_step = (micro_step % accum_steps == 0) or is_last_micro
                    grad_norm = None
                    if did_optimizer_step:
                        grad_norm, grads_finite = clip_grad_norm_fp64(trainable_params, args.grad_clip)
                        if not grads_finite:
                            optimizer.zero_grad(set_to_none=True)
                            raise RuntimeError("non-finite gradients")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        global_step += 1
                    metrics = {
                        "micro_step": micro_step,
                        "step": global_step,
                        "epoch": epoch,
                        "sample_pos": batch_start + 1,
                        "batch_size": len(items),
                        "grad_accum_steps": accum_steps,
                        "did_optimizer_step": did_optimizer_step,
                        "img_ids": [info["img_id"] for info in batch["sample_info"]],
                        "num_instances": sum(info["num_instances"] for info in batch["sample_info"]),
                        "loss": float(total.detach().cpu()),
                        "loss_components": {k: float(v.detach().mean().cpu()) for k, v in losses.items() if torch.is_tensor(v)},
                        "grad_norm": grad_norm,
                        "cuda_allocated_gb": round(torch.cuda.memory_allocated(device) / (1024 ** 3), 3),
                        "cuda_peak_gb": round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 3),
                        "skipped": skipped,
                        "elapsed_sec": round(time.time() - start_time, 3),
                    }
                    log_f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
                    log_f.flush()
                    if did_optimizer_step and (global_step == 1 or global_step % args.log_interval == 0):
                        print(json.dumps({"event": "train_step", **metrics}, ensure_ascii=False), flush=True)
                    if did_optimizer_step and args.save_interval > 0 and global_step % args.save_interval == 0:
                        save_checkpoint(work_dir, model, optimizer, global_step, epoch, args, metrics)
                except (RuntimeError, OSError) as exc:
                    optimizer.zero_grad(set_to_none=True)
                    if "out of memory" in str(exc).lower():
                        torch.cuda.empty_cache()
                    if not args.skip_sample_errors:
                        raise
                    skipped += 1
                    err = {"event": "batch_skip", "epoch": epoch, "sample_pos": batch_start + 1, "indices": batch_indices, "error": repr(exc), "skipped": skipped}
                    log_f.write(json.dumps(err, ensure_ascii=False) + "\n")
                    log_f.flush()
                    print(json.dumps(err, ensure_ascii=False), flush=True)
            if args.max_steps and global_step >= args.max_steps:
                break
    final_metrics = {"step": global_step, "epoch": min(args.max_epochs, epoch), "skipped": skipped, "elapsed_sec": round(time.time() - start_time, 3)}
    if global_step > 0:
        save_checkpoint(work_dir, model, optimizer, global_step, min(args.max_epochs, epoch), args, final_metrics)
    print(json.dumps({"event": "train_done", **final_metrics}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
