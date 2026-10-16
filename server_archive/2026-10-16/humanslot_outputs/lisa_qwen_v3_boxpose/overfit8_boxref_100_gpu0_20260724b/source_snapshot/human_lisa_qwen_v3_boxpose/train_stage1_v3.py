#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset, audit_annotation_file
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone
from human_lisa_qwen_v2.collator_v2 import Stage1V2Collator
from human_lisa_qwen_v3_boxpose.modeling_stage1_v3 import DEFAULT_LOSS_WEIGHTS_V3, Stage1V3Model
from human_lisa_qwen_v2.schema_v2 import PROMPT_V2, SPECIAL_TOKENS_V2
from human_lisa_qwen_v3_boxpose.training_v3 import (
    assert_finite_gradients,
    assert_finite_optimizer_state,
    build_cosine_scheduler,
    build_constant_scheduler,
    build_optimizer,
    accumulation_group_state,
    clip_global_gradient,
    gradient_statistics,
    move_to_device,
    optimizer_steps_for_micro_plan,
    save_training_state,
    scalar_tensors,
)


MODEL_INPUT_KEYS = {
    "input_ids", "attention_mask", "labels", "language_loss_weights", "language_task_ids",
    "pixel_values", "image_grid_thw", "mm_token_type_ids", "num_persons", "box_targets",
    "pose_targets", "pose_areas", "seg_targets", "pose_person_indices", "seg_person_indices",
    "original_sizes",
}


class EpochPermutationSampler(Sampler[int]):
    def __init__(self, size: int, seed: int) -> None:
        self.size = int(size); self.seed = int(seed); self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())

    def __len__(self) -> int:
        return self.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-ann", default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json")
    parser.add_argument("--train-image-root", default="/root/autodl-tmp/coco/train2017")
    parser.add_argument("--val-ann", default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json")
    parser.add_argument("--val-image-root", default="/root/autodl-tmp/coco/val2017")
    parser.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.5-4B")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-from", default=None, help="Resume only from a completed epoch checkpoint in output-dir")
    parser.add_argument("--run-kind", choices=["overfit", "smoke", "full"], default="smoke")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-optimizer-steps", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--horizontal-flip-prob", type=float, default=0.5)
    parser.add_argument("--visual-lr", type=float, default=5e-6)
    parser.add_argument("--language-lr", type=float, default=5e-6)
    parser.add_argument("--token-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--scheduler-type", choices=["cosine", "constant"], default="cosine")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-mode", choices=["all-linear", "attention-only"], default="all-linear")
    parser.add_argument("--vision-layers", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--person-query-layers", type=int, default=3)
    parser.add_argument("--pose-decoder-layers", type=int, default=2)
    parser.add_argument("--max-people", type=int, default=25)
    parser.add_argument("--query-objectness-weight", type=float, default=1.0)
    parser.add_argument("--query-bbox-l1-weight", type=float, default=5.0)
    parser.add_argument("--query-giou-weight", type=float, default=2.0)
    parser.add_argument("--query-aux-weight", type=float, default=0.5)
    parser.add_argument("--pose-l1-weight", type=float, default=10.0)
    parser.add_argument("--pose-oks-weight", type=float, default=4.0)
    parser.add_argument("--pose-visibility-weight", type=float, default=1.0)
    parser.add_argument("--pose-aux-weight", type=float, default=0.5)
    parser.add_argument("--mask-bce-weight", type=float, default=2.0)
    parser.add_argument("--mask-dice-weight", type=float, default=5.0)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-epoch-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stop-after-epoch", type=int, default=0, help="Controlled epoch-boundary pause; excluded from method manifest")
    parser.add_argument("--skip-save", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def reproducibility_source_files() -> List[Path]:
    files = [
        path for path in (REPO_ROOT / "human_lisa_qwen_v3_boxpose").rglob("*")
        if path.is_file() and path.suffix in {".py", ".md"}
        and "__pycache__" not in path.parts
    ]
    files.extend(
        REPO_ROOT / relative
        for relative in [
            "human_lisa_qwen_v1/data_v1.py",
            "human_lisa_qwen_v1/modeling.py",
            "human_lisa_qwen_v1/inference_v1.py",
            "human_lisa_qwen_v1/official_eval_v1.py",
            "human_lisa_qwen_v2/collator_v2.py",
            "human_lisa_qwen_v2/heads_v2.py",
            "human_lisa_qwen_v2/schema_v2.py",
        ]
    )
    unique = {path.resolve() for path in files}
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"reproducibility source files missing: {missing}")
    return sorted(unique, key=lambda path: str(path.relative_to(REPO_ROOT)))


def source_fingerprints(files: List[Path]) -> Dict[str, str]:
    return {
        str(path.relative_to(REPO_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def snapshot_sources(files: List[Path], destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"source snapshot already exists: {destination}")
    for source in files:
        target = destination / source.relative_to(REPO_ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def text_hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    for candidate in [getattr(config, "text_config", None), config]:
        if candidate is not None and getattr(candidate, "hidden_size", None):
            return int(candidate.hidden_size)
    return int(model.get_input_embeddings().embedding_dim)


def add_and_initialize_tokens(processor: Any, model: torch.nn.Module) -> Dict[str, int]:
    tokenizer = processor.tokenizer
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    tokens = existing + [token for token in SPECIAL_TOKENS_V2 if token not in existing]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    embedding_rows = int(model.get_input_embeddings().num_embeddings)
    if len(tokenizer) > embedding_rows:
        model.resize_token_embeddings(len(tokenizer))
    token_ids = {
        name: int(tokenizer.convert_tokens_to_ids(token))
        for name, token in {"person": "<person>", "pose": "<pose>", "seg": "<seg>"}.items()
    }
    if len(set(token_ids.values())) != 3 or any(value < 0 for value in token_ids.values()):
        raise RuntimeError(f"invalid V2 token ids: {token_ids}")
    seeds = {"person": "person", "pose": "pose", "seg": "segmentation"}
    input_weight = model.get_input_embeddings().weight
    output_module = model.get_output_embeddings()
    output_weight = output_module.weight if output_module is not None else None
    with torch.no_grad():
        for name, token_id in token_ids.items():
            source_ids = tokenizer(seeds[name], add_special_tokens=False)["input_ids"]
            source = torch.tensor(source_ids, device=input_weight.device, dtype=torch.long)
            input_weight[token_id].copy_(input_weight.index_select(0, source).mean(dim=0))
            if output_weight is not None and output_weight.data_ptr() != input_weight.data_ptr():
                output_weight[token_id].copy_(output_weight.index_select(0, source.to(output_weight.device)).mean(dim=0))
    return token_ids


def loss_weights(args: argparse.Namespace) -> Dict[str, float]:
    result = dict(DEFAULT_LOSS_WEIGHTS_V3)
    result.update(
        {
            "query_objectness": args.query_objectness_weight,
            "query_bbox_l1": args.query_bbox_l1_weight,
            "query_giou": args.query_giou_weight,
            "query_aux": args.query_aux_weight,
            "pose_l1": args.pose_l1_weight,
            "pose_oks": args.pose_oks_weight,
            "pose_visibility": args.pose_visibility_weight,
            "pose_aux": args.pose_aux_weight,
            "mask_bce": args.mask_bce_weight,
            "mask_dice": args.mask_dice_weight,
        }
    )
    if any(not math.isfinite(value) or value < 0.0 for value in result.values()):
        raise ValueError(f"invalid loss weights: {result}")
    return result


def audit_lora_coverage(model: torch.nn.Module, mode: str) -> Dict[str, Any]:
    patterns = {
        "visual_attention_blocks": re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.attn\."),
        "visual_mlp_blocks": re.compile(r"(?:^|\.)visual\.blocks\.(\d+)\.mlp\."),
        "language_self_attention_layers": re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.self_attn\."),
        "language_linear_attention_layers": re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.linear_attn\."),
        "language_mlp_layers": re.compile(r"(?:^|\.)language_model\.layers\.(\d+)\.mlp\."),
    }
    covered = {name: set() for name in patterns}
    lora_parameters = 0
    lora_tensors = 0
    for parameter_name, parameter in model.named_parameters():
        if "lora_" not in parameter_name:
            continue
        lora_parameters += parameter.numel()
        lora_tensors += 1
        for family, pattern in patterns.items():
            match = pattern.search(parameter_name)
            if match:
                covered[family].add(int(match.group(1)))
    coverage = {name: sorted(values) for name, values in covered.items()}
    expected_counts = {
        "all-linear": {
            "visual_attention_blocks": 24,
            "visual_mlp_blocks": 24,
            "language_self_attention_layers": 8,
            "language_linear_attention_layers": 24,
            "language_mlp_layers": 32,
        },
        "attention-only": {
            "visual_attention_blocks": 24,
            "visual_mlp_blocks": 0,
            "language_self_attention_layers": 8,
            "language_linear_attention_layers": 0,
            "language_mlp_layers": 0,
        },
    }[mode]
    actual_counts = {name: len(values) for name, values in coverage.items()}
    if actual_counts != expected_counts:
        raise RuntimeError(
            f"LoRA coverage mismatch for mode={mode}: actual={actual_counts}, "
            f"expected={expected_counts}"
        )
    return {
        "mode": mode,
        "lora_parameters": lora_parameters,
        "lora_tensors": lora_tensors,
        "counts": actual_counts,
        "layers": coverage,
    }


def build_model(args: argparse.Namespace, processor: Any, dtype: torch.dtype) -> Stage1V3Model:
    from peft import LoraConfig, cast_mixed_precision_params, get_peft_model

    base = load_qwen_vl_backbone(args.model, dtype)
    token_ids = add_and_initialize_tokens(processor, base)
    target_modules = "all-linear" if args.lora_target_mode == "all-linear" else [
        "q_proj", "k_proj", "v_proj", "o_proj", "attn.qkv", "attn.proj"
    ]
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
        trainable_token_indices=list(token_ids.values()),
        ensure_weight_tying=True,
    )
    base = get_peft_model(base, config)
    cast_mixed_precision_params(base, dtype)
    base.config.use_cache = False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    return Stage1V3Model(
        base_model=base,
        text_hidden_size=text_hidden_size(base),
        token_ids=token_ids,
        vision_layer_indices=args.vision_layers,
        decoder_dim=args.decoder_dim,
        decoder_heads=args.decoder_heads,
        person_query_layers=args.person_query_layers,
        pose_decoder_layers=args.pose_decoder_layers,
        max_people=args.max_people,
        loss_weights=loss_weights(args),
    )


def load_trainable_checkpoint(
    args: argparse.Namespace,
    processor: Any,
    dtype: torch.dtype,
    checkpoint: Path,
) -> tuple[Stage1V3Model, Dict[str, Any]]:
    from peft import PeftModel, cast_mixed_precision_params

    checkpoint_manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    base = load_qwen_vl_backbone(args.model, dtype)
    token_ids = add_and_initialize_tokens(processor, base)
    expected_token_ids = {name: int(value) for name, value in checkpoint_manifest["token_ids"].items()}
    if token_ids != expected_token_ids:
        raise RuntimeError(f"resume special-token ids differ: current={token_ids}, checkpoint={expected_token_ids}")
    base = PeftModel.from_pretrained(base, checkpoint / "base_or_adapter", is_trainable=True)
    cast_mixed_precision_params(base, dtype)
    base.config.use_cache = False
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()
    state = torch.load(checkpoint / "stage1_v3_task_modules.pt", map_location="cpu", weights_only=False)
    wrapper = Stage1V3Model(
        base_model=base,
        text_hidden_size=text_hidden_size(base),
        token_ids=token_ids,
        vision_layer_indices=state["vision_layer_indices"],
        decoder_dim=int(state["decoder_dim"]),
        decoder_heads=int(state["decoder_heads"]),
        person_query_layers=int(state["person_query_layers"]),
        pose_decoder_layers=int(state["pose_decoder_layers"]),
        visual_merge_size=int(state["visual_merge_size"]),
        max_people=int(state["max_people"]),
        loss_weights=state["loss_weights"],
    )
    wrapper.load_task_modules(str(checkpoint / "stage1_v3_task_modules.pt"))
    return wrapper, checkpoint_manifest


def save_checkpoint(
    checkpoint: Path,
    *,
    model: Stage1V3Model,
    processor: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: Dict[str, Any],
    manifest: Dict[str, Any],
) -> None:
    if checkpoint.exists():
        raise FileExistsError(f"checkpoint already exists: {checkpoint}")
    temporary = checkpoint.with_name(checkpoint.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    model.base_model.save_pretrained(temporary / "base_or_adapter")
    model.save_task_modules(str(temporary))
    processor.save_pretrained(temporary / "processor")
    (temporary / "manifest.json").write_text(json.dumps({**manifest, "train_state": train_state}, indent=2), encoding="utf-8")
    save_training_state(temporary / "training_state.pt", optimizer=optimizer, scheduler=scheduler, train_state=train_state)
    temporary.rename(checkpoint)


def main() -> None:
    args = parse_args()
    if args.max_length != 5000:
        raise ValueError("Stage1 V3 requires max_length=5000 with truncation disabled")
    if args.max_people != 25:
        raise ValueError("Stage1 V3 uses the audited K=25 person-query capacity")
    if args.batch_size <= 0 or args.grad_accum_steps <= 0 or args.epochs <= 0:
        raise ValueError("batch size, accumulation, and epochs must be positive")
    if args.stop_after_epoch < 0 or args.stop_after_epoch > args.epochs:
        raise ValueError("stop-after-epoch must be 0 or within the planned epoch count")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage1 V3 flat-pose training requires CUDA")
    set_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    resume_checkpoint = Path(args.resume_from).resolve() if args.resume_from else None
    if resume_checkpoint is None:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f"refusing to overwrite non-empty run: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        if not resume_checkpoint.is_dir():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_checkpoint}")
        if resume_checkpoint.parent != output_dir:
            raise ValueError("resume checkpoint must be an immediate child of output-dir")
        if not output_dir.is_dir() or not (output_dir / "run_manifest.json").is_file():
            raise FileNotFoundError("resume output-dir is missing its original run_manifest.json")
    event_file = output_dir / "events.jsonl"

    def emit(payload: Dict[str, Any]) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        print(line, flush=True)
        with event_file.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    processor_source = resume_checkpoint / "processor" if resume_checkpoint is not None else args.model
    processor = AutoProcessor.from_pretrained(processor_source, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if resume_checkpoint is None:
        model = build_model(args, processor, dtype)
        checkpoint_manifest = None
    else:
        model, checkpoint_manifest = load_trainable_checkpoint(args, processor, dtype, resume_checkpoint)
    lora_coverage = audit_lora_coverage(model, args.lora_target_mode)
    device = torch.device("cuda")
    model.to(device).train()

    dataset = CocoUniHumanV1Dataset(
        args.train_ann,
        args.train_image_root,
        training=True,
        horizontal_flip_prob=args.horizontal_flip_prob,
        max_samples=args.max_samples,
    )
    sampler = EpochPermutationSampler(len(dataset), args.seed)
    collator = Stage1V2Collator(
        processor=processor,
        token_ids=model.token_ids,
        prompt=PROMPT_V2,
        max_length=args.max_length,
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
    planned_micro_steps = micro_steps_per_epoch * args.epochs
    target_micro_steps = planned_micro_steps
    if args.max_optimizer_steps > 0:
        if args.grad_accum_steps > 1 and micro_steps_per_epoch % args.grad_accum_steps:
            raise ValueError("max-optimizer-steps with cross-epoch partial accumulation is unsupported")
        target_micro_steps = min(target_micro_steps, args.max_optimizer_steps * args.grad_accum_steps)
    optimizer_steps_per_epoch = optimizer_steps_for_micro_plan(
        micro_steps_per_epoch, micro_steps_per_epoch, args.grad_accum_steps
    )
    total_optimizer_steps = optimizer_steps_for_micro_plan(
        micro_steps_per_epoch, target_micro_steps, args.grad_accum_steps
    )
    optimizer, optimizer_manifest = build_optimizer(
        model,
        visual_lr=args.visual_lr,
        language_lr=args.language_lr,
        token_lr=args.token_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scheduler_builder = build_cosine_scheduler if args.scheduler_type == "cosine" else build_constant_scheduler
    scheduler = scheduler_builder(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_ratio=args.warmup_ratio,
    )
    trainable_total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    source_files = reproducibility_source_files()
    source_sha256 = source_fingerprints(source_files)
    manifest = {
        "version": "stage1_v3_flatpose",
        "run_kind": args.run_kind,
        "source_sha256": source_sha256,
        "model": args.model,
        "train_ann": args.train_ann,
        "train_image_root": args.train_image_root,
        "val_ann": args.val_ann,
        "val_image_root": args.val_image_root,
        "dataset_size": len(dataset),
        "annotation_audit": {
            "train": audit_annotation_file(args.train_ann),
            "val": audit_annotation_file(args.val_ann),
            "segmentation_policy": "train supervision follows available COCO person masks; val segm AP uses raw COCO polygons, matching HumanQueryNet evaluate_det_segm",
        },
        "architecture": {
            "person_binding": "K=25 image-conditioned person queries with Hungarian assignment",
            "text_binding": "generated/teacher-forced person records align to matched queries by person box",
            "box": "native autoregressive integer coordinates plus auxiliary query box/objectness",
            "confidence": "Hungarian-matched query objectness ranks native text boxes for standard COCO AP; constant-score AP is also reported",
            "pose": "matched person query plus <person>/<pose> hidden states; direct COCO-17 coordinate MLP with no box, ROI, heatmap, or jitter input",
            "seg": "matched person query plus <seg> hidden state and LISA/PixelLM-style dynamic mask decoder",
            "missing_labels": "task-specific masking; missing pose/seg labels are never negatives",
            "vision_features": {"raw_qwen_layers": list(args.vision_layers), "spatial_pyramid": 3},
            "decoder_dim": args.decoder_dim,
            "decoder_heads": args.decoder_heads,
            "person_query_layers": args.person_query_layers,
            "pose_decoder_layers": args.pose_decoder_layers,
            "max_people": args.max_people,
            "single_forward_backward": True,
            "references": {
                "HumanQueryNet": "COCO-UniHuman official 7b5f57b",
                "GroupPose": "127fe0e (query and loss protocol reference; pose head intentionally simplified)",
                "LISA": "3cb2d43",
                "PixelLM": "ebf0792",
                "PaDT": "912a8ae",
            },
        },
        "prompt": PROMPT_V2,
        "token_ids": model.token_ids,
        "max_length": args.max_length,
        "truncation": False,
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "epochs": args.epochs,
        "micro_steps_per_epoch": micro_steps_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "target_micro_steps": target_micro_steps,
        "total_optimizer_steps": total_optimizer_steps,
        "gradient_accumulation_scope": "reset_and_step_at_each_epoch_boundary",
        "optimizer": optimizer_manifest,
        "loss_weights": model.loss_weights,
        "global_grad_clip": args.grad_clip,
        "warmup_ratio": args.warmup_ratio,
        "scheduler_type": args.scheduler_type,
        "bf16": args.bf16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "horizontal_flip_prob": args.horizontal_flip_prob,
        "lora": {
            "rank": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_mode": args.lora_target_mode,
            "all_visual_blocks": True,
            "coverage": lora_coverage,
        },
        "trainable_total": trainable_total,
        "seed": args.seed,
    }
    resume_payload = None
    if resume_checkpoint is None:
        snapshot_sources(source_files, output_dir / "source_snapshot")
        (output_dir / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        emit({"event": "training_plan", **manifest})
    else:
        original_manifest = {key: value for key, value in checkpoint_manifest.items() if key != "train_state"}
        differing = sorted(
            key for key in set(manifest) | set(original_manifest)
            if manifest.get(key) != original_manifest.get(key)
        )
        if differing:
            raise RuntimeError(f"resume manifest mismatch in fields: {differing}")
        resume_payload = torch.load(resume_checkpoint / "training_state.pt", map_location="cpu", weights_only=False)
        resume_state = resume_payload.get("train_state", {})
        if "epoch_completed" not in resume_state:
            raise RuntimeError("resume is allowed only from a completed epoch checkpoint")
        start_epoch = int(resume_state["epoch_completed"])
        expected_micro_step = start_epoch * micro_steps_per_epoch
        expected_optimizer_step = start_epoch * optimizer_steps_per_epoch
        if int(resume_state.get("micro_step", -1)) != expected_micro_step:
            raise RuntimeError(
                f"epoch-boundary resume mismatch: micro_step={resume_state.get('micro_step')} "
                f"expected={expected_micro_step}"
            )
        if int(resume_state.get("optimizer_step", -1)) != expected_optimizer_step:
            raise RuntimeError(
                f"epoch-boundary resume mismatch: optimizer_step={resume_state.get('optimizer_step')} "
                f"expected={expected_optimizer_step}"
            )
        optimizer.load_state_dict(resume_payload["optimizer"])
        scheduler.load_state_dict(resume_payload["scheduler"])
        torch.set_rng_state(resume_payload["torch_rng"])
        if resume_payload.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(resume_payload["cuda_rng"])
        if resume_payload.get("python_rng") is not None:
            random.setstate(resume_payload["python_rng"])
        if resume_payload.get("numpy_rng") is not None:
            np.random.set_state(resume_payload["numpy_rng"])
        emit({"event": "resume", "checkpoint": str(resume_checkpoint), "resume_state": resume_state})

    optimizer.zero_grad(set_to_none=True)
    if resume_payload is None:
        start_epoch = 0
        micro_step = 0
        optimizer_step = 0
        samples_seen = 0
    else:
        state = resume_payload["train_state"]
        micro_step = int(state["micro_step"])
        optimizer_step = int(state["optimizer_step"])
        samples_seen = int(state["samples_seen"])
    resume_micro_step = micro_step
    resume_samples_seen = samples_seen
    if micro_step >= target_micro_steps:
        raise RuntimeError("resume checkpoint has already completed the requested training plan")
    started = time.perf_counter()
    loss_window: Dict[str, float] = {}
    window_micro = 0
    first_gradient_gate_passed = False

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        batches_in_epoch = 0
        for raw_batch in loader:
            if micro_step >= target_micro_steps:
                break
            batch_started = time.perf_counter()
            batches_in_epoch += 1
            micro_step += 1
            batch_samples = len(raw_batch["sample_ids"])
            samples_seen += batch_samples
            group_divisor, optimizer_boundary = accumulation_group_state(
                batches_in_epoch=batches_in_epoch,
                micro_step=micro_step,
                micro_steps_per_epoch=micro_steps_per_epoch,
                target_micro_steps=target_micro_steps,
                grad_accum_steps=args.grad_accum_steps,
            )
            model_batch = move_to_device(
                {name: value for name, value in raw_batch.items() if name in MODEL_INPUT_KEYS},
                device,
            )
            with torch.autocast("cuda", dtype=dtype, enabled=args.bf16):
                outputs = model(**model_batch)
                loss = outputs["loss"]
            if not torch.isfinite(loss.detach()):
                raise FloatingPointError(f"non-finite loss at micro_step={micro_step}")
            (loss / float(group_divisor)).backward()
            assert_finite_gradients(model)
            for name, value in scalar_tensors(outputs).items():
                loss_window[name] = loss_window.get(name, 0.0) + value
            window_micro += 1

            if not optimizer_boundary:
                continue
            preclip_stats = gradient_statistics(model)
            if not first_gradient_gate_passed:
                invalid = {
                    family: values
                    for family, values in preclip_stats.items()
                    if not values["finite"] or values["nonzero"] <= 0
                }
                if invalid:
                    raise RuntimeError(f"first-step gradient gate failed: {invalid}")
                first_gradient_gate_passed = True
            preclip_norm = clip_global_gradient(model, args.grad_clip)
            optimizer.step()
            assert_finite_optimizer_state(optimizer, model)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            train_state = {
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "epoch_index": epoch,
                "batches_in_epoch": batches_in_epoch,
                "samples_seen": samples_seen,
            }
            should_log = optimizer_step == 1 or optimizer_step % args.log_every == 0
            if should_log:
                elapsed = time.perf_counter() - started
                averaged = {name: value / max(1, window_micro) for name, value in loss_window.items()}
                emit(
                    {
                        "event": "train",
                        **train_state,
                        "elapsed_seconds": elapsed,
                        "samples_per_second": (samples_seen - resume_samples_seen) / max(elapsed, 1e-6),
                        "eta_seconds": elapsed / max(1, micro_step - resume_micro_step) * (target_micro_steps - micro_step),
                        "batch_seconds": time.perf_counter() - batch_started,
                        "learning_rates": {group["group_name"]: group["lr"] for group in optimizer.param_groups},
                        "preclip_global_grad_norm": preclip_norm,
                        "gradient_families": preclip_stats,
                        "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1048576, 1),
                        "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1048576, 1),
                        "max_sequence_tokens": int(raw_batch["sequence_lengths"].max().item()),
                        **averaged,
                    }
                )
                loss_window = {}; window_micro = 0
            if not args.skip_save and args.save_every > 0 and optimizer_step % args.save_every == 0:
                save_checkpoint(
                    output_dir / f"checkpoint-step-{optimizer_step}",
                    model=model,
                    processor=processor,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    train_state=train_state,
                    manifest=manifest,
                )
        completed_epoch = batches_in_epoch == micro_steps_per_epoch
        if completed_epoch and not args.skip_save and args.save_epoch_checkpoints:
            save_checkpoint(
                output_dir / f"epoch-{epoch + 1}-step-{optimizer_step}",
                model=model,
                processor=processor,
                optimizer=optimizer,
                scheduler=scheduler,
                train_state={
                    "micro_step": micro_step,
                    "optimizer_step": optimizer_step,
                    "epoch_index": epoch,
                    "epoch_completed": epoch + 1,
                    "samples_seen": samples_seen,
                },
                manifest=manifest,
            )
        emit({"event": "epoch_end", "epoch": epoch + 1, "completed": completed_epoch, "micro_step": micro_step, "optimizer_step": optimizer_step})
        if args.stop_after_epoch and epoch + 1 >= args.stop_after_epoch and micro_step < target_micro_steps:
            emit({"event": "planned_pause", "epoch_completed": epoch + 1, "micro_step": micro_step, "optimizer_step": optimizer_step})
            break
        if micro_step >= target_micro_steps:
            break

    final_state = {
        "micro_step": micro_step,
        "optimizer_step": optimizer_step,
        "samples_seen": samples_seen,
        "elapsed_seconds": time.perf_counter() - started,
        "completed_full_plan": micro_step == planned_micro_steps,
    }
    # A completed epoch checkpoint already contains identical trainable and
    # optimizer state. Keep a final checkpoint only for runs that disabled
    # epoch saves or stopped inside an epoch.
    if not args.skip_save and (not completed_epoch or not args.save_epoch_checkpoints):
        save_checkpoint(
            output_dir / f"final-step-{optimizer_step}",
            model=model,
            processor=processor,
            optimizer=optimizer,
            scheduler=scheduler,
            train_state=final_state,
            manifest=manifest,
        )
    emit({"event": "completed", **final_state})


if __name__ == "__main__":
    main()
