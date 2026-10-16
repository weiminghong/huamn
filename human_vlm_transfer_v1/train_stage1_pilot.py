from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoProcessor

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset, audit_annotation_file
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone
from human_lisa_qwen_v4.token_rows_v4 import SelectiveTokenRows

from .data import Stage1TransferCollator
from .modeling import Stage1TransferModel


ROUTE_TOKENS = {"pose": "<pose>", "seg": "<seg>"}
ANCHOR_TOKENS = {
    "human": "<human_task>",
    "box": "<box_task>",
    "mask": "<mask_task>",
    "pose": "<pose_task>",
    "gender": "<gender_task>",
    "age": "<age_task>",
}
SEMANTIC_SOURCES = {
    "<pose>": "pose",
    "<seg>": "segmentation",
    "<human_task>": "human person",
    "<box_task>": "person detection bounding box",
    "<mask_task>": "person segmentation mask",
    "<pose_task>": "human pose keypoints",
    "<gender_task>": "gender attribute",
    "<age_task>": "age attribute",
}
PROMPT = """Analyze every visible non-crowd real person.
The internal task anchors are:
<human_task> <box_task> <mask_task> <pose_task> <gender_task> <age_task>
Return one JSON object and nothing else.
Use this exact schema:
{"persons":[{"person_id":1,"box":[x1,y1,x2,y2],"pose":"<pose>","seg":"<seg>","gender":0,"age":31}]}
Rules:
- Order people left-to-right by body-box center, then top-to-bottom.
- person_id must be contiguous starting at 1.
- box is the full visible body box as four integers from 0 to 1000.
- pose must be exactly "<pose>" and seg must be exactly "<seg>".
- gender is 0 for male, 1 for female, and -1 only when unknown.
- age is apparent age from 0 to 120, or -1 when unknown.
- If no person is visible, return exactly {"persons":[]}.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-ann",
        default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json",
    )
    parser.add_argument(
        "--train-image-root",
        default="/root/autodl-tmp/coco/train2017",
    )
    parser.add_argument(
        "--model",
        default="/root/autodl-tmp/model/Qwen3.5-4B",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--select-task-rich", action="store_true")
    parser.add_argument("--vision-start-block", type=int, default=16)
    parser.add_argument("--vision-layers", default="7,15,23")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--language-lr", type=float, default=2e-5)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--merger-lr", type=float, default=1e-5)
    parser.add_argument("--token-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-compact", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), sort_keys=True) + "\n")
        handle.flush()


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_processor(model_path: str) -> Tuple[Any, Dict[str, int], Dict[str, int]]:
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    all_tokens = list(ROUTE_TOKENS.values()) + list(ANCHOR_TOKENS.values())
    existing = list(
        getattr(processor.tokenizer, "additional_special_tokens", []) or []
    )
    additions = [token for token in all_tokens if token not in existing]
    if additions:
        processor.tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + additions}
        )
    route_ids = {
        name: int(processor.tokenizer.convert_tokens_to_ids(token))
        for name, token in ROUTE_TOKENS.items()
    }
    anchor_ids = {
        name: int(processor.tokenizer.convert_tokens_to_ids(token))
        for name, token in ANCHOR_TOKENS.items()
    }
    ids = list(route_ids.values()) + list(anchor_ids.values())
    if len(set(ids)) != len(ids) or min(ids) < 0:
        raise RuntimeError(f"invalid special token ids: {ids}")
    return processor, route_ids, anchor_ids


def discover_lora_targets(
    model: nn.Module, *, vision_start_block: int
) -> Tuple[List[str], List[str]]:
    visual = []
    language = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        padded = f".{name}."
        match = re.search(r"\.visual\.blocks\.(\d+)\.", padded)
        if match is not None and int(match.group(1)) >= int(vision_start_block):
            visual.append(name)
            continue
        if ".language_model.layers." in padded:
            language.append(name)
    if not visual:
        raise RuntimeError("no visual LoRA targets discovered")
    if not language:
        raise RuntimeError("no language LoRA targets discovered")
    return sorted(visual), sorted(language)


def resolve_vision_module(model: nn.Module) -> Tuple[str, nn.Module]:
    candidates = []
    for name, module in model.named_modules():
        if (
            hasattr(module, "blocks")
            and hasattr(module, "merger")
            and hasattr(module, "patch_embed")
            and len(getattr(module, "blocks")) == 24
        ):
            candidates.append((name, module))
    if not candidates:
        raise RuntimeError("Qwen vision module was not found")
    candidates.sort(key=lambda item: (item[0].count("."), len(item[0])))
    return candidates[0]


def add_special_rows(
    processor: Any,
    model: nn.Module,
    route_ids: Mapping[str, int],
    anchor_ids: Mapping[str, int],
) -> Tuple[SelectiveTokenRows, List[Dict[str, Any]]]:
    tokenizer = processor.tokenizer
    if len(tokenizer) > int(model.get_input_embeddings().num_embeddings):
        model.resize_token_embeddings(len(tokenizer))
    ordered_tokens = list(ROUTE_TOKENS.values()) + list(ANCHOR_TOKENS.values())
    ordered_ids = [
        int(tokenizer.convert_tokens_to_ids(token)) for token in ordered_tokens
    ]
    token_rows = SelectiveTokenRows(
        ordered_ids,
        int(model.get_input_embeddings().embedding_dim),
    )
    token_rows.attach(model)
    source_ids = [
        [
            int(value)
            for value in tokenizer(
                SEMANTIC_SOURCES[token],
                add_special_tokens=False,
            )["input_ids"]
        ]
        for token in ordered_tokens
    ]
    audit = token_rows.initialize_from_source_ids(model, source_ids)
    return token_rows, audit


def build_model(
    args: argparse.Namespace,
    processor: Any,
    route_ids: Mapping[str, int],
    anchor_ids: Mapping[str, int],
) -> Tuple[Stage1TransferModel, Dict[str, Any]]:
    base = load_qwen_vl_backbone(args.model, torch.bfloat16)
    if len(processor.tokenizer) > int(base.get_input_embeddings().num_embeddings):
        base.resize_token_embeddings(len(processor.tokenizer))
    base.requires_grad_(False)
    visual_targets, language_targets = discover_lora_targets(
        base, vision_start_block=args.vision_start_block
    )
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=visual_targets + language_targets,
        modules_to_save=[],
    )
    base = get_peft_model(base, lora)
    if hasattr(base, "config"):
        base.config.use_cache = False
    if hasattr(base, "gradient_checkpointing_enable"):
        base.gradient_checkpointing_enable()
    if hasattr(base, "enable_input_require_grads"):
        base.enable_input_require_grads()

    token_rows, token_audit = add_special_rows(
        processor, base, route_ids, anchor_ids
    )
    merger_names = []
    merger_parameters = 0
    for name, parameter in base.named_parameters():
        if ".visual.merger." in f".{name}.":
            parameter.requires_grad_(True)
            merger_names.append(name)
            merger_parameters += parameter.numel()
    if not merger_names:
        raise RuntimeError("visual merger parameters were not found")

    vision_name, vision_module = resolve_vision_module(base)
    config = base.config
    vision_config = config.vision_config
    text_config = config.text_config
    layer_indices = [
        int(value.strip()) for value in args.vision_layers.split(",")
        if value.strip()
    ]
    model = Stage1TransferModel(
        base_model=base,
        vision_module=vision_module,
        vision_layer_indices=layer_indices,
        vision_hidden_size=int(vision_config.hidden_size),
        text_hidden_size=int(text_config.hidden_size),
        merge_size=int(vision_config.spatial_merge_size),
        anchor_token_ids=anchor_ids,
        num_queries=args.num_queries,
        decoder_dim=args.decoder_dim,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        text_task_weights={
            "structure": 0.25,
            "box": 1.0,
            "pose_route": 0.25,
            "seg_route": 0.25,
            "gender": 1.0,
            "age": 1.0,
        },
    )
    model.token_rows = token_rows
    return model, {
        "vision_module": vision_name,
        "vision_layers": layer_indices,
        "visual_lora_targets": len(visual_targets),
        "language_lora_targets": len(language_targets),
        "visual_lora_examples": visual_targets[:8],
        "language_lora_examples": language_targets[:8],
        "merger_parameters": merger_parameters,
        "merger_examples": merger_names[:8],
        "token_initialization": token_audit,
    }


def family_for_parameter(name: str) -> str:
    padded = f".{name}."
    if name.startswith("base_model.") and "lora_" in name:
        if ".visual." in padded:
            return "visual_lora"
        if ".language_model." in padded:
            return "language_lora"
    if name.startswith("base_model.") and ".visual.merger." in padded:
        return "merger"
    if name.startswith("token_rows."):
        return "token_rows"
    if name.startswith("vision_module."):
        return "alias"
    if not name.startswith("base_model."):
        return "heads"
    return "unexpected"


def build_optimizer(
    model: nn.Module,
    args: argparse.Namespace,
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    rates = {
        "visual_lora": args.vision_lr,
        "language_lora": args.language_lr,
        "merger": args.merger_lr,
        "token_rows": args.token_lr,
        "heads": args.head_lr,
    }
    grouped: Dict[Tuple[str, bool], List[torch.Tensor]] = {}
    counts = {name: 0 for name in rates}
    seen = set()
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        family = family_for_parameter(name)
        if family == "alias":
            continue
        if family == "unexpected":
            unexpected.append(name)
            continue
        counts[family] += parameter.numel()
        no_decay = (
            parameter.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or "embedding" in name.lower()
        )
        grouped.setdefault((family, no_decay), []).append(parameter)
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected[:20]}")
    missing = [name for name, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(f"missing trainable parameter families: {missing}")
    parameter_groups = []
    for (family, no_decay), parameters in grouped.items():
        parameter_groups.append(
            {
                "params": parameters,
                "lr": float(rates[family]),
                "weight_decay": 0.0 if no_decay else args.weight_decay,
                "family": family,
            }
        )
    optimizer = torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, {
        "trainable_total": int(sum(counts.values())),
        "trainable_by_family": {key: int(value) for key, value in counts.items()},
        "learning_rates": rates,
    }


def gradient_norms(model: nn.Module) -> Dict[str, float]:
    totals: Dict[str, torch.Tensor] = {}
    seen = set()
    for name, parameter in model.named_parameters():
        if parameter.grad is None or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        family = family_for_parameter(name)
        if family in {"alias", "unexpected"}:
            continue
        value = parameter.grad.detach().float().square().sum()
        totals[family] = totals.get(family, value.new_zeros(())) + value
    return {
        family: float(value.sqrt().cpu().item())
        for family, value in totals.items()
    }


def select_task_rich_indices(dataset: Dataset, maximum: int) -> List[int]:
    selected = []
    scan_limit = min(len(dataset), max(256, maximum * 16))
    for index in range(scan_limit):
        sample = dataset[index]
        instances = sample["instances"]
        if (
            instances
            and any(item["pose_valid"] for item in instances)
            and any(item["seg_valid"] for item in instances)
        ):
            selected.append(index)
            if len(selected) >= maximum:
                break
    if not selected:
        raise RuntimeError("no task-rich records found in scan window")
    return selected


def compact_checkpoint(model: Stage1TransferModel, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model.base_model.save_pretrained(output / "adapter")
    task_state = {
        name: value.detach().cpu()
        for name, value in model.state_dict().items()
        if not name.startswith("base_model.")
        and not name.startswith("vision_module.")
    }
    torch.save(task_state, output / "task_modules.pt")
    merger_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.base_model.named_parameters()
        if ".visual.merger." in f".{name}."
    }
    torch.save(merger_state, output / "visual_merger.pt")


def scalar_metrics(outputs: Mapping[str, Any]) -> Dict[str, float]:
    return {
        key: float(value.detach().float().cpu().item())
        for key, value in outputs.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    events_path = output / "events.jsonl"
    started = time.time()
    set_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    processor, route_ids, anchor_ids = prepare_processor(args.model)
    dataset = CocoUniHumanV1Dataset(
        args.train_ann,
        args.train_image_root,
        training=True,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples if args.max_samples > 0 else None,
        decode_eval_segmentation=False,
    )
    dataset_for_loader: Dataset = dataset
    selected_indices: List[int] | None = None
    if args.select_task_rich:
        selected_indices = select_task_rich_indices(
            dataset, max(args.batch_size * args.steps, args.batch_size)
        )
        dataset_for_loader = Subset(dataset, selected_indices)

    collator = Stage1TransferCollator(
        processor=processor,
        token_ids=dict(route_ids),
        prompt=PROMPT,
        max_length=5000,
    )
    loader = DataLoader(
        dataset_for_loader,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collator,
        drop_last=False,
    )
    model, model_manifest = build_model(
        args, processor, route_ids, anchor_ids
    )
    model.to(device)
    optimizer, optimizer_manifest = build_optimizer(model, args)

    manifest = {
        "experiment_id": output.name,
        "classification": "diagnostic_stage1_architecture_pilot",
        "method_version": "stage1_transfer_v1_pose_roi_only",
        "method_changes": [
            "box-ROI pose features with 32x24 normalized crops",
            "ground-truth box conditioning for pose training",
            "normalized and reweighted ROI heatmap cross-entropy",
        ],
        "objective": (
            "Validate joint Qwen vision/LLM adaptation with shared Human Queries, "
            "pre-merger spatial features, box-conditioned pose, and dynamic masks."
        ),
        "command_args": vars(args),
        "working_dir": str(Path.cwd()),
        "model_path": args.model,
        "model_config_sha256": sha256_file(str(Path(args.model) / "config.json")),
        "train_annotation": args.train_ann,
        "train_annotation_sha256": sha256_file(args.train_ann),
        "train_image_root": args.train_image_root,
        "annotation_audit": audit_annotation_file(args.train_ann),
        "selected_indices": selected_indices,
        "route_token_ids": route_ids,
        "anchor_token_ids": anchor_ids,
        "prompt": PROMPT,
        "model_manifest": model_manifest,
        "optimizer_manifest": optimizer_manifest,
        "seed": args.seed,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "cuda_device": torch.cuda.get_device_name(device),
        "started_unix": started,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    iterator = iter(loader)
    completed_steps = 0
    observed_gradient_families = set()
    while completed_steps < args.steps:
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        tensor_keys = (
            "input_ids",
            "attention_mask",
            "labels",
            "text_task_ids",
            "image_grid_thw",
            "pixel_values",
            "mm_token_type_ids",
        )
        tensors = {
            key: batch[key].to(device, non_blocking=True)
            for key in tensor_keys
            if key in batch
        }
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            outputs = model(
                input_ids=tensors["input_ids"],
                attention_mask=tensors["attention_mask"],
                labels=tensors["labels"],
                text_task_ids=tensors["text_task_ids"],
                image_grid_thw=tensors["image_grid_thw"],
                pixel_values=tensors["pixel_values"],
                mm_token_type_ids=tensors.get("mm_token_type_ids"),
                dense_targets=batch["dense_targets"],
            )
        loss = outputs["loss"]
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {completed_steps + 1}")
        loss.backward()
        gradients = gradient_norms(model)
        observed_gradient_families.update(
            name for name, value in gradients.items() if value > 0.0
        )
        torch.nn.utils.clip_grad_norm_(
            [
                parameter
                for parameter in model.parameters()
                if parameter.requires_grad and parameter.grad is not None
            ],
            args.grad_clip,
        )
        optimizer.step()
        completed_steps += 1
        event = {
            "event": "optimizer_step",
            "step": completed_steps,
            "elapsed_seconds": time.time() - started,
            "sample_ids": batch["sample_ids"],
            "matched_persons": outputs["matched_persons"],
            "mask_labels": outputs["mask_labels"],
            "pose_labels": outputs["pose_labels"],
            "feature_shapes": outputs["feature_shapes"],
            "gradient_norms": gradients,
            "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        }
        event.update(scalar_metrics(outputs))
        append_jsonl(events_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)

    required_gradients = {
        "visual_lora",
        "language_lora",
        "merger",
        "token_rows",
        "heads",
    }
    missing_gradients = sorted(required_gradients - observed_gradient_families)
    if missing_gradients:
        raise RuntimeError(
            f"required gradient families were never observed: {missing_gradients}"
        )
    if args.save_compact:
        compact_checkpoint(model, output / "checkpoint-final")
    status = {
        "status": "completed",
        "steps": completed_steps,
        "elapsed_seconds": time.time() - started,
        "peak_cuda_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "observed_gradient_families": sorted(observed_gradient_families),
    }
    (output / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
