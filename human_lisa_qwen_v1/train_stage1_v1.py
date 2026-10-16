#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.collator_v1 import Stage1Collator
from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.modeling import (
    apply_lora_if_requested,
    load_qwen_vl_backbone,
)
from human_lisa_qwen_v1.modeling_stage1_v1 import Stage1V1Model
from human_lisa_qwen_v1.schema_v1 import PROMPT_V1, SPECIAL_TOKENS_V1
from human_lisa_qwen_v1.training_v1 import (
    accumulation_group_size,
    assert_finite_optimizer_state,
    assert_finite_training_state,
    build_optimizer,
    clip_gradient_families,
    build_warmup_constant_scheduler,
    build_warmup_cosine_scheduler,
    load_training_state,
    move_to_device,
    prepare_mixed_precision_parameters,
    force_visual_lora_fp32_updates,
    save_training_state,
    set_trainable_families,
    restrict_visual_lora_to_blocks,
    install_visual_gradient_sanitizer,
    should_optimizer_step,
)


MODEL_INPUT_KEYS = {
    "input_ids",
    "attention_mask",
    "labels",
    "language_loss_weights",
    "pixel_values",
    "image_grid_thw",
    "mm_token_type_ids",
    "box_targets",
    "pose_targets",
    "pose_areas",
    "seg_targets",
    "box_person_indices",
    "pose_person_indices",
    "seg_person_indices",
    "original_sizes",
}


class EpochPermutationSampler(Sampler[int]):
    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size)
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, start_index: int = 0) -> None:
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        if not 0 <= self.start_index <= self.size:
            raise ValueError("sampler start_index is outside the dataset")

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        permutation = torch.randperm(self.size, generator=generator).tolist()
        return iter(permutation[self.start_index :])

    def __len__(self) -> int:
        return self.size - self.start_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ann",
        default="/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json",
    )
    parser.add_argument(
        "--train-image-root",
        default="/share2/program/weiminghong/coco/train2017",
    )
    parser.add_argument(
        "--val-ann",
        default="/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json",
    )
    parser.add_argument(
        "--val-image-root",
        default="/share2/program/weiminghong/coco/val2017",
    )
    parser.add_argument(
        "--model",
        default="/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct",
    )
    parser.add_argument(
        "--output-dir",
        default="/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/train",
    )
    parser.add_argument("--resume-dir", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--stop-after-micro-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--missing-attribute-loss-weight", type=float, default=0.0)
    parser.add_argument("--ce-loss-weight", type=float, default=1.0)
    parser.add_argument("--box-l1-loss-weight", type=float, default=5.0)
    parser.add_argument("--box-giou-loss-weight", type=float, default=2.0)
    parser.add_argument("--box-quality-loss-weight", type=float, default=1.0)
    parser.add_argument("--pose-l1-loss-weight", type=float, default=5.0)
    parser.add_argument("--pose-oks-loss-weight", type=float, default=1.5)
    parser.add_argument("--pose-heatmap-loss-weight", type=float, default=0.0)
    parser.add_argument("--pose-visibility-loss-weight", type=float, default=1.0)
    parser.add_argument("--mask-bce-loss-weight", type=float, default=8.0)
    parser.add_argument("--mask-dice-loss-weight", type=float, default=5.0)
    parser.add_argument(
        "--trainable-families",
        nargs="+",
        choices=["visual", "language", "box", "pose", "mask"],
        default=["visual", "language", "box", "pose", "mask"],
    )
    parser.add_argument(
        "--reset-optimizer-on-resume", action="store_true"
    )
    parser.add_argument(
        "--reset-pose-decoder-on-resume", action="store_true"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--base-lr", type=float, default=2e-5)
    parser.add_argument("--visual-lr", type=float, default=None)
    parser.add_argument("--language-lr", type=float, default=None)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--box-lr", type=float, default=None)
    parser.add_argument("--pose-lr", type=float, default=None)
    parser.add_argument("--mask-lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument(
        "--scheduler-type",
        choices=["cosine", "constant"],
        default="cosine",
    )
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--task-backbone-grad-scale",
        type=float,
        default=0.1,
    )
    parser.add_argument("--visual-lora-start-block", type=int, default=16)
    parser.add_argument(
        "--pose-box-jitter",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--pose-box-shift-factor", type=float, default=0.16)
    parser.add_argument("--pose-box-shift-prob", type=float, default=0.3)
    parser.add_argument("--pose-box-scale-min", type=float, default=0.5)
    parser.add_argument("--pose-box-scale-max", type=float, default=1.5)
    parser.add_argument("--pose-box-scale-prob", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-embed-tokens", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lora-lm-head", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--visual-lora-fp32-updates", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--split-ce-task-backward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--probe-every", type=int, default=500)
    parser.add_argument("--probe-count", type=int, default=4)
    parser.add_argument("--probe-max-new-tokens", type=int, default=5000)
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--dry-run-data", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="")
    return parser.parse_args()


def loss_weights_from_args(args: argparse.Namespace) -> Dict[str, float]:
    weights = {
        "ce": float(args.ce_loss_weight),
        "box_l1": float(args.box_l1_loss_weight),
        "box_giou": float(args.box_giou_loss_weight),
        "box_quality": float(args.box_quality_loss_weight),
        "pose_l1": float(args.pose_l1_loss_weight),
        "pose_oks": float(args.pose_oks_loss_weight),
        "pose_heatmap": float(args.pose_heatmap_loss_weight),
        "pose_visibility": float(args.pose_visibility_loss_weight),
        "mask_bce": float(args.mask_bce_loss_weight),
        "mask_dice": float(args.mask_dice_loss_weight),
    }
    invalid = {
        name: value
        for name, value in weights.items()
        if not math.isfinite(value) or value < 0.0
    }
    if invalid:
        raise ValueError(
            f"loss weights must be finite and non-negative: {invalid}"
        )
    return weights


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hidden_size_from_config(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for value in [
        getattr(config, "text_config", None),
        getattr(config, "llm_config", None),
        config,
    ]:
        if value is not None and getattr(value, "hidden_size", None):
            return int(value.hidden_size)
    return int(model.get_input_embeddings().embedding_dim)


def add_special_tokens(
    processor: Any, model: torch.nn.Module
) -> Dict[str, int]:
    tokenizer = processor.tokenizer
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    added = [token for token in SPECIAL_TOKENS_V1 if token not in existing]
    if added:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + added}
        )
        embedding_rows = int(model.get_input_embeddings().num_embeddings)
        if len(tokenizer) > embedding_rows:
            model.resize_token_embeddings(len(tokenizer))
    token_ids = {
        name: int(tokenizer.convert_tokens_to_ids(token))
        for name, token in {
            "box": "<box>",
            "pose": "<pose>",
            "seg": "<seg>",
        }.items()
    }
    if len(set(token_ids.values())) != 3 or any(
        value < 0 for value in token_ids.values()
    ):
        raise RuntimeError(f"invalid special-token ids: {token_ids}")
    return token_ids


def lora_targets(*, include_embed_tokens: bool = True, include_lm_head: bool = True) -> List[str]:
    targets = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "attn.qkv",
        "attn.proj",
    ]
    if include_embed_tokens:
        targets.append("embed_tokens")
    if include_lm_head:
        targets.append("lm_head")
    return targets


def trainable_manifest(model: torch.nn.Module) -> Dict[str, Any]:
    groups: Dict[str, int] = {
        "visual_lora": 0,
        "language_lora": 0,
        "task_heads": 0,
        "embeddings_or_lm_head": 0,
        "other": 0,
    }
    examples: Dict[str, List[str]] = {name: [] for name in groups}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith(("box_head.", "pose_decoder.", "mask_decoder.")):
            group = "task_heads"
        elif "lora_" in name and ".visual." in name:
            group = "visual_lora"
        elif "lora_" in name:
            group = "language_lora"
        elif "embed_tokens" in name or "lm_head" in name:
            group = "embeddings_or_lm_head"
        else:
            group = "other"
        groups[group] += int(parameter.numel())
        if len(examples[group]) < 5:
            examples[group].append(name)
    if sum(groups.values()) <= 0:
        raise RuntimeError("model has no trainable parameters")
    return {
        "trainable_total": sum(groups.values()),
        "trainable_by_group": groups,
        "trainable_examples": examples,
    }


def make_base_and_wrapper(
    args: argparse.Namespace,
    processor: Any,
    dtype: torch.dtype,
) -> Stage1V1Model:
    base = load_qwen_vl_backbone(args.model, dtype)
    token_ids = add_special_tokens(processor, base)
    pose_decoder_type = "direct_joint_query"
    if args.resume_dir:
        task_state = torch.load(
            Path(args.resume_dir) / "stage1_task_heads.pt",
            map_location="cpu",
            weights_only=False,
        )
        pose_decoder_type = str(
            task_state.get("pose_decoder_type", "coordinate")
        )
        from peft import PeftModel

        adapter_dir = Path(args.resume_dir) / "base_or_adapter"
        base = PeftModel.from_pretrained(
            base, adapter_dir, is_trainable=True
        )
    else:
        base = apply_lora_if_requested(
            base,
            True,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=lora_targets(
                include_embed_tokens=args.lora_embed_tokens,
                include_lm_head=args.lora_lm_head,
            ),
            modules_to_save=[],
        )
    for candidate in [
        base,
        getattr(base, "base_model", None),
        getattr(getattr(base, "base_model", None), "model", None),
    ]:
        config = getattr(candidate, "config", None)
        if config is not None:
            config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()
    config = getattr(base, "config", None)
    image_token_id = int(getattr(config, "image_token_id", 151655))
    wrapper = Stage1V1Model(
        base_model=base,
        hidden_size=hidden_size_from_config(base),
        token_ids=token_ids,
        image_token_id=image_token_id,
        decoder_dim=args.decoder_dim,
        decoder_heads=args.decoder_heads,
        pose_decoder_type=pose_decoder_type,
        loss_weights=loss_weights_from_args(args),
        visual_merge_size=2,
        task_backbone_grad_scale=args.task_backbone_grad_scale,
        pose_box_jitter_enabled=args.pose_box_jitter,
        pose_box_shift_factor=args.pose_box_shift_factor,
        pose_box_shift_prob=args.pose_box_shift_prob,
        pose_box_scale_range=(
            args.pose_box_scale_min, args.pose_box_scale_max
        ),
        pose_box_scale_prob=args.pose_box_scale_prob,
    )
    if args.resume_dir:
        wrapper.load_task_heads(
            str(Path(args.resume_dir) / "stage1_task_heads.pt"),
            restore_training_config=False,
            load_pose_decoder=not args.reset_pose_decoder_on_resume,
        )
    return wrapper


def scalar_outputs(outputs: Dict[str, Any]) -> Dict[str, float]:
    result = {}
    for name, value in outputs.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            result[name] = float(value.detach().float().cpu())
    return result


def save_checkpoint(
    checkpoint_dir: Path,
    *,
    model: Stage1V1Model,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    if checkpoint_dir.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint_dir}")
    temporary = checkpoint_dir.with_name(checkpoint_dir.name + ".tmp")
    if temporary.exists():
        if temporary.parent != checkpoint_dir.parent:
            raise RuntimeError("temporary checkpoint escaped output directory")
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.save_task_heads(str(temporary))
    model.base_model.save_pretrained(
        temporary / "base_or_adapter", save_embedding_layers=False
    )
    processor.save_pretrained(temporary / "processor")
    save_training_state(
        temporary / "training_state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_state=train_state,
        include_model_state=False,
    )
    with (temporary / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {**manifest, "train_state": train_state},
            handle,
            ensure_ascii=False,
            indent=2,
        )
    temporary.rename(checkpoint_dir)


def run_decode_probe(
    *,
    model: Stage1V1Model,
    processor: Any,
    samples: List[Dict[str, Any]],
    output_dir: Path,
    optimizer_step: int,
    max_new_tokens: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Any]:
    from human_lisa_qwen_v1.inference_v1 import (
        infer_samples,
        summarize_inference_audit,
    )
    from human_lisa_qwen_v1.visualize_v1 import render_probe_overlays

    was_training = model.training
    model.eval()
    results = infer_samples(
        model,
        processor,
        samples,
        max_new_tokens=max_new_tokens,
        dtype=dtype,
        device=device,
    )
    if was_training:
        model.train()
    probe_dir = output_dir / "decode_probes"
    probe_dir.mkdir(parents=True, exist_ok=True)
    probe_path = probe_dir / f"step-{optimizer_step}.json"
    with probe_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    overlay_summary = render_probe_overlays(
        samples,
        results,
        probe_dir,
        step=optimizer_step,
    )
    audit = summarize_inference_audit(results)
    count_errors = [
        abs(len(result.get("records", [])) - len(sample["instances"]))
        for result, sample in zip(results, samples)
    ]
    raw_count_errors = [
        abs(
            int(result.get("raw_record_count", len(result.get("records", []))))
            - len(sample["instances"])
        )
        for result, sample in zip(results, samples)
    ]
    pose_count_errors = [
        abs(
            sum("keypoints" in record for record in result.get("records", []))
            - sum(bool(instance.get("pose_valid", False)) for instance in sample["instances"])
        )
        for result, sample in zip(results, samples)
    ]
    seg_count_errors = [
        abs(
            sum("segmentation" in record for record in result.get("records", []))
            - sum(bool(instance.get("seg_valid", False)) for instance in sample["instances"])
        )
        for result, sample in zip(results, samples)
    ]
    duplicate_images = sum(
        int("duplicate_boxes" in result.get("health", {}).get("errors", []))
        for result in results
    )
    suppressed_records = sum(
        len(result.get("duplicate_suppression", {}).get("suppressed_records", []))
        for result in results
    )
    summary = {
        "probe_step": int(optimizer_step),
        "probe_images": len(results),
        "schema_ok_rate": audit["schema_ok_rate"],
        "raw_schema_ok_rate": audit["raw_schema_ok_rate"],
        "person_id_repair_image_rate": audit["person_id_repair_image_rate"],
        "person_id_repairs": audit["person_id_repairs"],
        "health_ok_rate": audit["health_ok_rate"],
        "eos_rate": audit["eos_rate"],
        "raw_person_count_mae": sum(raw_count_errors) / max(1, len(raw_count_errors)),
        "person_count_mae": sum(count_errors) / max(1, len(count_errors)),
        "pose_count_mae": sum(pose_count_errors) / max(1, len(pose_count_errors)),
        "seg_count_mae": sum(seg_count_errors) / max(1, len(seg_count_errors)),
        "raw_duplicate_box_image_rate": duplicate_images / max(1, len(results)),
        "nms_suppressed_records": int(suppressed_records),
        "probe_path": str(probe_path),
        "overlay_files": overlay_summary["files"],
        "overlay_persons": overlay_summary["persons"],
    }
    print(json.dumps({"decode_probe": summary}, ensure_ascii=False), flush=True)
    return summary


def data_dry_run(args: argparse.Namespace) -> None:
    dataset = CocoUniHumanV1Dataset(
        args.train_ann,
        args.train_image_root,
        training=True,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples or 4,
    )
    records = []
    for index in range(min(len(dataset), 4)):
        sample = dataset[index]
        records.append(
            {
                "image_id": sample["id"],
                "image_size": [sample["width"], sample["height"]],
                "persons": len(sample["instances"]),
                "box": sum(1 for _ in sample["instances"]),
                "pose": sum(
                    int(instance["pose_valid"])
                    for instance in sample["instances"]
                ),
                "seg": sum(
                    int(instance["seg_valid"])
                    for instance in sample["instances"]
                ),
                "gender": sum(
                    int(instance["gender_valid"])
                    for instance in sample["instances"]
                ),
                "age": sum(
                    int(instance["age_valid"])
                    for instance in sample["instances"]
                ),
                "skipped": sample["skipped"],
            }
        )
    print(json.dumps({"data_dry_run": records}, indent=2), flush=True)


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.max_length != 5000:
        raise ValueError("the stable Stage1 configuration requires max_length=5000")
    if args.reset_pose_decoder_on_resume and not args.resume_dir:
        raise ValueError(
            "reset_pose_decoder_on_resume requires resume_dir"
        )
    if args.grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be positive")
    set_seed(args.seed)
    if args.dry_run_data:
        data_dry_run(args)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("Stage1 training requires CUDA")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = make_base_and_wrapper(args, processor, dtype)
    trainable_family_counts = set_trainable_families(
        model, set(args.trainable_families)
    )
    if "visual" in set(args.trainable_families):
        visual_lora_block_selection = restrict_visual_lora_to_blocks(
            model, args.visual_lora_start_block
        )
        trainable_family_counts["visual"] = visual_lora_block_selection["trainable"]
    else:
        visual_lora_block_selection = {
            "start_block": int(args.visual_lora_start_block),
            "trainable": 0,
            "frozen": 0,
            "skipped": True,
        }
    parameter_precision = (
        prepare_mixed_precision_parameters(model, dtype)
        if dtype == torch.bfloat16
        else {"trainable_fp32": sum(
            int(parameter.numel())
            for parameter in model.parameters()
            if parameter.requires_grad
        )}
    )
    visual_lora_fp32_modules = (
        force_visual_lora_fp32_updates(model)
        if (
            args.bf16
            and args.visual_lora_fp32_updates
            and "visual" in set(args.trainable_families)
        )
        else 0
    )
    device = torch.device("cuda")
    model.to(device)
    model.train()
    if "visual" in set(args.trainable_families):
        visual_gradient_sanitizer = install_visual_gradient_sanitizer(model)
    else:
        visual_gradient_sanitizer = {
            "hooks": 0,
            "drain": lambda: {"events": 0, "parameters": []},
        }
    trainable = trainable_manifest(model)

    dataset = CocoUniHumanV1Dataset(
        args.train_ann,
        args.train_image_root,
        training=True,
        horizontal_flip_prob=args.horizontal_flip_prob,
        max_samples=args.max_samples,
    )
    sampler = EpochPermutationSampler(len(dataset), args.seed)
    probe_samples: List[Dict[str, Any]] = []
    if args.probe_every > 0 and args.probe_count > 0:
        original_flip_probability = dataset.horizontal_flip_prob
        dataset.horizontal_flip_prob = 0.0
        probe_samples = [
            dataset[index]
            for index in range(min(args.probe_count, len(dataset)))
        ]
        dataset.horizontal_flip_prob = original_flip_probability
    collator = Stage1Collator(
        processor=processor,
        token_ids=model.token_ids,
        prompt=PROMPT_V1,
        max_length=args.max_length,
        missing_attribute_loss_weight=args.missing_attribute_loss_weight,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )
    micro_steps_per_epoch = len(loader)
    epoch_micro_steps = int(
        math.ceil(float(args.epochs) * micro_steps_per_epoch)
    )
    if args.max_optimizer_steps > 0:
        total_micro_steps = min(
            epoch_micro_steps,
            int(args.max_optimizer_steps) * args.grad_accum_steps,
        )
    else:
        total_micro_steps = epoch_micro_steps
    total_optimizer_steps = int(
        math.ceil(total_micro_steps / args.grad_accum_steps)
    )
    optimizer, optimizer_summary = build_optimizer(
        model,
        base_lr=args.base_lr,
        visual_lr=args.visual_lr,
        language_lr=args.language_lr,
        head_lr=args.head_lr,
        box_lr=args.box_lr,
        pose_lr=args.pose_lr,
        mask_lr=args.mask_lr,
        weight_decay=args.weight_decay,
    )
    scheduler_builder = (
        build_warmup_constant_scheduler
        if args.scheduler_type == "constant"
        else build_warmup_cosine_scheduler
    )
    scheduler = scheduler_builder(
        optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
    )
    train_state = {
        "micro_step": 0,
        "optimizer_step": 0,
        "epoch": 0,
        "micro_in_epoch": 0,
        "samples_seen": 0,
    }
    if args.resume_dir and not args.reset_optimizer_on_resume:
        train_state = load_training_state(
            Path(args.resume_dir) / "training_state.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            load_model_state=False,
        )

    manifest = {
        "model": args.model,
        "architecture": {
            "person_records": "variable_length",
            "box_head": "token_state_box_quality",
            "pose_head": (
                "direct_joint_query_reference_sampling"
                if model.pose_decoder_type == "direct_joint_query"
                else model.pose_decoder_type
            ),
            "pose_decoder_type": model.pose_decoder_type,
            "pose_reference": (
                {
                    "train": "predicted_box_detached",
                    "eval": "predicted_box_detached",
                    "sampling_points": 8,
                    "refinement_layers": int(model.pose_decoder.num_layers),
                }
                if model.pose_decoder_type == "direct_joint_query"
                else None
            ),
            "pose_heatmap_size": int(
                getattr(model.pose_decoder, "heatmap_size", 0)
            ),
            "mask_head": "token_state_pixel_decoder",
            "record_nms": {"near_duplicate_iou": 0.90},
            "pose_roi_enabled": bool(model.pose_roi_enabled),
            "pose_box_jitter": {
                "enabled": model.pose_box_jitter_enabled,
                "shift_factor": model.pose_box_shift_factor,
                "shift_prob": model.pose_box_shift_prob,
                "scale_range": list(model.pose_box_scale_range),
                "scale_prob": model.pose_box_scale_prob,
            },
        },
        "train_ann": args.train_ann,
        "pose_joint_reference": (
            "trainable_coco_skeleton_logits"
            if model.pose_decoder_type == "coordinate"
            else None
        ),
        "train_image_root": args.train_image_root,
        "val_ann": args.val_ann,
        "val_image_root": args.val_image_root,
        "dataset_size": len(dataset),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "max_length": args.max_length,
        "missing_attribute_loss_weight": args.missing_attribute_loss_weight,
        "trainable_families": list(args.trainable_families),
        "trainable_family_parameters": trainable_family_counts,
        "parameter_precision": parameter_precision,
        "resume_optimizer_state": bool(args.resume_dir and not args.reset_optimizer_on_resume),
        "reset_pose_decoder_on_resume": args.reset_pose_decoder_on_resume,
        "epochs": args.epochs,
        "total_micro_steps": total_micro_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "base_lr": args.base_lr,
        "visual_lr": (
            args.base_lr if args.visual_lr is None else args.visual_lr
        ),
        "language_lr": (
            args.base_lr if args.language_lr is None else args.language_lr
        ),
        "head_lr": args.head_lr,
        "box_lr": args.head_lr if args.box_lr is None else args.box_lr,
        "pose_lr": args.head_lr if args.pose_lr is None else args.pose_lr,
        "mask_lr": args.head_lr if args.mask_lr is None else args.mask_lr,
        "warmup_ratio": args.warmup_ratio,
        "scheduler_type": args.scheduler_type,
        "grad_clip": args.grad_clip,
        "task_backbone_grad_scale": args.task_backbone_grad_scale,
        "visual_lora_blocks": visual_lora_block_selection,
        "weight_decay": args.weight_decay,
        "bf16": args.bf16,
        "visual_lora_fp32_updates": {
            "enabled": bool(args.bf16 and args.visual_lora_fp32_updates),
            "modules": int(visual_lora_fp32_modules),
        },
        "split_ce_task_backward": bool(args.split_ce_task_backward),
        "visual_gradient_sanitizer": {"enabled": True, "hooks": int(visual_gradient_sanitizer["hooks"])},
        "seed": args.seed,
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "lora": {
            "rank": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "targets": lora_targets(
                include_embed_tokens=args.lora_embed_tokens,
                include_lm_head=args.lora_lm_head,
            ),
            "embed_tokens_lora": bool(args.lora_embed_tokens),
            "lm_head_lora": bool(args.lora_lm_head),
        },
        "loss_weights": dict(model.loss_weights),
        "decoder_dim": args.decoder_dim,
        "decoder_heads": args.decoder_heads,
        "token_ids": model.token_ids,
        "image_token_id": model.image_token_id,
        "optimizer": optimizer_summary,
        **trainable,
    }
    with (output_dir / "run_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"training_plan": manifest}, ensure_ascii=False), flush=True)

    optimizer.zero_grad(set_to_none=True)
    micro_step = int(train_state["micro_step"])
    optimizer_step = int(train_state["optimizer_step"])
    start_epoch = int(train_state["epoch"])
    start_in_epoch = int(train_state["micro_in_epoch"])
    accumulated: Dict[str, float] = {}
    accumulated_micro = 0
    accumulated_samples = 0
    log_accumulated: Dict[str, float] = {}
    log_micro_count = 0
    sanitization_window: Dict[str, Any] = {"events": 0, "parameters": set()}

    def record_sanitization(stats: Dict[str, Any]) -> None:
        sanitization_window["events"] += int(stats["events"])
        sanitization_window["parameters"].update(stats["parameters"])

    for epoch in range(start_epoch, int(math.ceil(args.epochs))):
        sampler.set_epoch(
            epoch, start_in_epoch if epoch == start_epoch else 0
        )
        consumed_in_epoch = start_in_epoch if epoch == start_epoch else 0
        for raw_batch in loader:
            if micro_step >= total_micro_steps or (
                args.stop_after_micro_steps > 0
                and micro_step >= args.stop_after_micro_steps
            ):
                break
            micro_step += 1
            batch_sample_count = len(raw_batch.get("sample_ids", []))
            if batch_sample_count <= 0:
                batch_sample_count = int(raw_batch["input_ids"].shape[0])
            consumed_in_epoch += batch_sample_count
            group_divisor = accumulation_group_size(
                micro_step, total_micro_steps, args.grad_accum_steps
            )
            model_batch = {
                name: value
                for name, value in raw_batch.items()
                if name in MODEL_INPUT_KEYS
            }
            model_batch = move_to_device(model_batch, device)
            with torch.autocast(
                "cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)
            ):
                outputs = model(**model_batch)
                task_component = sum(
                    outputs[f"weighted_{name}"]
                    for name in (
                        "box_l1", "box_giou", "box_quality",
                        "pose_l1", "pose_oks", "pose_heatmap", "pose_visibility",
                        "mask_bce", "mask_dice",
                    )
                )
                loss = outputs["loss"]
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(
                    f"non-finite loss before backward at micro_step={micro_step}"
                )
            if args.split_ce_task_backward and args.ce_loss_weight != 0.0:
                (task_component / float(group_divisor)).backward()
                sanitized_task = visual_gradient_sanitizer["drain"]()
                record_sanitization(sanitized_task)
                assert_finite_training_state(task_component, model)
                with torch.autocast(
                    "cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)
                ):
                    ce_outputs = model(**model_batch)
                    ce_component = ce_outputs["ce_loss"] * float(args.ce_loss_weight)
                if not torch.isfinite(ce_component.detach()):
                    raise FloatingPointError(
                        f"non-finite CE loss before backward at micro_step={micro_step}"
                    )
                (ce_component / float(group_divisor)).backward()
                sanitized_ce = visual_gradient_sanitizer["drain"]()
                record_sanitization(sanitized_ce)
                outputs["ce_loss"] = ce_outputs["ce_loss"]
                outputs["loss"] = task_component.detach() + ce_component
                loss = outputs["loss"]
            else:
                (loss / float(group_divisor)).backward()
                sanitized_loss = visual_gradient_sanitizer["drain"]()
                record_sanitization(sanitized_loss)
            assert_finite_training_state(loss, model)
            for name, value in scalar_outputs(outputs).items():
                accumulated[name] = accumulated.get(name, 0.0) + value
            accumulated_micro += 1
            accumulated_samples += batch_sample_count

            if should_optimizer_step(
                micro_step, total_micro_steps, args.grad_accum_steps
            ):
                grad_norms = clip_gradient_families(
                    model, max_norm=args.grad_clip
                )
                grad_norm_value = max(grad_norms.values(), default=0.0)
                assert_finite_training_state(loss, model)
                optimizer.step()
                assert_finite_optimizer_state(optimizer, model)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                train_state = {
                    "micro_step": micro_step,
                    "optimizer_step": optimizer_step,
                    "epoch": epoch,
                    "micro_in_epoch": consumed_in_epoch,
                    "samples_seen": int(train_state["samples_seen"])
                    + accumulated_samples,
                }
                for name, value in accumulated.items():
                    log_accumulated[name] = log_accumulated.get(name, 0.0) + value
                log_micro_count += accumulated_micro
                if (
                    optimizer_step == 1
                    or optimizer_step % args.log_every == 0
                ):
                    averaged = {
                        name: value / log_micro_count
                        for name, value in log_accumulated.items()
                    }
                    log = {
                        "micro_step": micro_step,
                        "optimizer_step": optimizer_step,
                        "epoch": epoch,
                        "micro_in_epoch": consumed_in_epoch,
                        "lr": [
                            group["lr"] for group in optimizer.param_groups
                        ],
                        "grad_norm": grad_norm_value,
                        "grad_norm_by_family": grad_norms,
                        "cuda_memory_allocated_mb": round(
                            torch.cuda.memory_allocated(device) / 1048576, 2
                        ),
                        "cuda_memory_reserved_mb": round(
                            torch.cuda.memory_reserved(device) / 1048576, 2
                        ),
                        "max_sequence_tokens": int(
                            raw_batch["sequence_lengths"].max().item()
                        ),
                        "log_window_micro_steps": int(log_micro_count),
                        "visual_grad_sanitization_events": int(
                            sanitization_window["events"]
                        ),
                        "visual_grad_sanitized_parameters": sorted(
                            sanitization_window["parameters"]
                        ),
                        **averaged,
                    }
                    print(json.dumps(log, ensure_ascii=False), flush=True)
                    log_accumulated = {}
                    log_micro_count = 0
                    sanitization_window = {"events": 0, "parameters": set()}
                accumulated = {}
                accumulated_micro = 0
                accumulated_samples = 0

                if (
                    probe_samples
                    and args.probe_every > 0
                    and optimizer_step % args.probe_every == 0
                ):
                    run_decode_probe(
                        model=model,
                        processor=processor,
                        samples=probe_samples,
                        output_dir=output_dir,
                        optimizer_step=optimizer_step,
                        max_new_tokens=args.probe_max_new_tokens,
                        dtype=dtype,
                        device=device,
                    )
                if (
                    not args.skip_save
                    and args.save_every > 0
                    and optimizer_step % args.save_every == 0
                ):
                    save_checkpoint(
                        output_dir / f"checkpoint-{optimizer_step}",
                        model=model,
                        processor=processor,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        train_state=train_state,
                        manifest=manifest,
                    )
        start_in_epoch = 0
        if micro_step >= total_micro_steps or (
            args.stop_after_micro_steps > 0
            and micro_step >= args.stop_after_micro_steps
        ):
            break

    if not args.skip_save:
        save_checkpoint(
            output_dir / f"final-step-{optimizer_step}",
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=train_state,
            manifest=manifest,
        )
    print(
        json.dumps(
            {"status": "completed", "train_state": train_state},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

