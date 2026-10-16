from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import random
import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from human_lisa_pose_stage1.train_overfit import (
    SelectiveTokenRows,
    build_optimizer,
    discover_lora_targets,
    resolve_vision_module,
)
from human_lisa_qwen_v1.modeling import load_qwen_vl_backbone

from .collator import QwenV5Collator
from .dataset import CanonicalPacketDataset
from .manifest import audit_manifests, iter_jsonl
from .modeling import Stage1V5Model
from .protocol import SPECIAL_TOKENS
from .sampling import BalancedPacketSampler, build_packet_index


SEMANTIC_SOURCES = {
    "<human>": "human person instance",
    "<pose>": "named human pose keypoints",
    "<seg>": "person segmentation mask",
    "<parse>": "human body part parsing",
}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_processor(model_path: str):
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True
    )
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    existing = list(
        getattr(tokenizer, "additional_special_tokens", []) or []
    )
    additions = [token for token in SPECIAL_TOKENS if token not in existing]
    if additions:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + additions}
        )
    ids = {
        token.strip("<>"): int(tokenizer.convert_tokens_to_ids(token))
        for token in SPECIAL_TOKENS
    }
    if len(set(ids.values())) != len(ids):
        raise RuntimeError(f"special token collision: {ids}")
    return processor, ids


def add_token_rows(processor, base, token_ids):
    if len(processor.tokenizer) > base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(processor.tokenizer))
    ids = [token_ids[token.strip("<>")] for token in SPECIAL_TOKENS]
    rows = SelectiveTokenRows(
        ids, int(base.get_input_embeddings().embedding_dim)
    )
    rows.attach(base)
    source_ids = [
        [
            int(value)
            for value in processor.tokenizer(
                SEMANTIC_SOURCES[token], add_special_tokens=False
            )["input_ids"]
        ]
        for token in SPECIAL_TOKENS
    ]
    audit = rows.initialize_from_source_ids(base, source_ids)
    return rows, audit


def build_model(args, processor, token_ids):
    base = load_qwen_vl_backbone(args.model, torch.bfloat16)
    if len(processor.tokenizer) > base.get_input_embeddings().num_embeddings:
        base.resize_token_embeddings(len(processor.tokenizer))
    base.requires_grad_(False)
    visual_targets, language_targets = discover_lora_targets(
        base, vision_start_block=0
    )
    base = get_peft_model(
        base,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=visual_targets + language_targets,
            modules_to_save=[],
        ),
    )
    base.config.use_cache = False
    base.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    base.enable_input_require_grads()
    token_rows, token_audit = add_token_rows(
        processor, base, token_ids
    )
    merger_names = []
    for name, parameter in base.named_parameters():
        if ".visual.merger." in f".{name}.":
            parameter.requires_grad_(True)
            parameter.data = parameter.data.float()
            merger_names.append(name)
    if not merger_names:
        raise RuntimeError("Qwen merger not found")
    vision_name, vision = resolve_vision_module(base)
    layers = [int(value) for value in args.vision_layers.split(",")]
    model = Stage1V5Model(
        base_model=base,
        vision_module=vision,
        vision_layer_indices=layers,
        vision_hidden_size=int(base.config.vision_config.hidden_size),
        text_hidden_size=int(base.config.text_config.hidden_size),
        merge_size=int(base.config.vision_config.spatial_merge_size),
        human_token_id=token_ids["human"],
        decoder_dim=args.decoder_dim,
        decoder_encoder_layers=args.mask_encoder_layers,
        decoder_layers=args.mask_decoder_layers,
        decoder_heads=args.decoder_heads,
    )
    model.token_rows = token_rows
    return model, {
        "vision_module": vision_name,
        "vision_layers": layers,
        "visual_lora_targets": len(visual_targets),
        "language_lora_targets": len(language_targets),
        "merger_parameters": sum(
            parameter.numel()
            for name, parameter in base.named_parameters()
            if ".visual.merger." in f".{name}."
        ),
        "token_initialization": token_audit,
        "dense_decoder": (
            "Transformers Mask2FormerPixelDecoder and "
            "Mask2FormerMaskedAttentionDecoder; Qwen query adaptation"
        ),
        "external_rgb_encoder": False,
    }


def save_checkpoint(model, optimizer, output: Path, cursor: Mapping[str, int]):
    checkpoint = output / f"checkpoint-step{cursor['global_step']:06d}"
    if checkpoint.is_dir():
        return checkpoint
    checkpoint.mkdir(exist_ok=False)
    model.base_model.save_pretrained(checkpoint / "adapter")
    torch.save(
        {
            name: value.detach().cpu()
            for name, value in model.state_dict().items()
            if not name.startswith("base_model.")
            and not name.startswith("vision_module.")
        },
        checkpoint / "v5_modules.pt",
    )
    torch.save(
        {"optimizer": optimizer.state_dict(), **dict(cursor)},
        checkpoint / "trainer_state.pt",
    )
    atomic_json(checkpoint / "cursor.json", cursor)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(Path(__file__).with_name("config_v5.json")))
    parser.add_argument("--manifests", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.5-4B")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--epoch-size", type=int, default=0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--vision-layers", default="7,15,23")
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--mask-encoder-layers", type=int, default=2)
    parser.add_argument("--mask-decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-heads", type=int, default=8)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--head-lr", type=float, default=1e-4)
    parser.add_argument("--language-lr", type=float, default=2e-5)
    parser.add_argument("--vision-lr", type=float, default=1e-5)
    parser.add_argument("--merger-lr", type=float, default=1e-5)
    parser.add_argument("--token-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--gpu-lock", default="/root/autodl-tmp/humanslot/.gpu0_exclusive.lock")
    return parser.parse_args()


def run(args) -> None:
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    audit = audit_manifests(args.manifests)
    if not audit["valid"]:
        raise RuntimeError("manifest audit failed")
    records = [
        record for path in args.manifests for record in iter_jsonl(path)
    ]
    index = build_packet_index(records)
    epoch_size = int(
        args.epoch_size or config["sampling"]["epoch_size"]
    )
    sampler = BalancedPacketSampler(
        index,
        config["sampling"]["task_weights"],
        epoch_size=epoch_size,
        seed=args.seed,
    )
    _, sampling_audit = sampler.epoch(0)
    preflight = {
        "schema": "human-multitask-v5-preflight",
        "manifest_audit": audit,
        "sampling_audit": sampling_audit.__dict__,
        "config": str(Path(args.config).resolve()),
        "manifests": [str(Path(path).resolve()) for path in args.manifests],
        "requested_parameter_scope": {
            "qwen_vision": "LoRA all blocks",
            "qwen_merger": "full",
            "qwen_llm": "attention/MLP LoRA",
            "dense": "Qwen bridge plus Mask2Former modules full",
            "original_qwen_weights": "frozen outside PEFT/merger",
        },
        "requested_epochs": args.epochs,
        "requested_epoch_size": epoch_size,
    }
    output = Path(args.output_dir)
    if args.preflight_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(output, preflight)
        print(json.dumps(preflight, sort_keys=True))
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for training")
    output.mkdir(parents=True, exist_ok=False)
    lock_handle = Path(args.gpu_lock).open("a+")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("GPU project lock is held by another run") from error

    set_seed(args.seed)
    processor, token_ids = prepare_processor(args.model)
    model, model_manifest = build_model(
        args, processor, token_ids
    )
    model.to("cuda")
    model.train()
    optimizer, optimizer_manifest = build_optimizer(model, args)
    total_updates = math.ceil(
        epoch_size / args.gradient_accumulation_steps
    ) * args.epochs
    manifest = {
        **preflight,
        "classification": "v5_full_multisource_joint_training",
        "total_optimizer_steps": total_updates,
        "model_manifest": model_manifest,
        "optimizer_manifest": optimizer_manifest,
        "token_ids": token_ids,
        "evaluation_cadence": "once_per_completed_epoch",
        "checkpoint_cadence": args.save_every,
        "started_unix": time.time(),
    }
    atomic_json(output / "manifest.json", manifest)
    processor.save_pretrained(output / "processor")
    shutil.copytree(Path(__file__).parent, output / "source")

    collator = QwenV5Collator(
        processor=processor,
        token_ids={
            "human": token_ids["human"],
            "seg": token_ids["seg"],
            "parse": token_ids["parse"],
        },
    )
    cursor = {"global_step": 0, "epoch": 0, "examples_seen": 0}
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        packets, epoch_audit = sampler.epoch(epoch)
        atomic_json(
            output / f"sampling_epoch{epoch + 1:02d}.json",
            epoch_audit.__dict__,
        )
        dataset = CanonicalPacketDataset(args.manifests, packets)
        loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collator,
        )
        for batch_index, batch in enumerate(loader):
            tensor_keys = (
                "input_ids", "attention_mask", "labels",
                "image_grid_thw", "pixel_values", "mm_token_type_ids",
            )
            tensors = {
                key: batch[key].to("cuda", non_blocking=True)
                for key in tensor_keys
                if key in batch
            }
            with torch.autocast("cuda", dtype=torch.bfloat16):
                values = model(
                    **tensors, dense_targets=batch["dense_targets"]
                )
                loss = values["loss"] / args.gradient_accumulation_steps
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite loss at {batch['sample_uids']}"
                )
            loss.backward()
            cursor["examples_seen"] += 1
            boundary = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(loader)
            )
            if not boundary:
                model._captured_vision.clear()
                continue
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            cursor["global_step"] += 1
            cursor["epoch"] = epoch
            event = {
                "step": cursor["global_step"],
                "epoch": epoch + (batch_index + 1) / len(loader),
                "examples_seen": cursor["examples_seen"],
                "task": batch["tasks"][0],
                "dataset": dataset.records[
                    packets[batch_index].record_index
                ]["dataset"],
                **{
                    key: float(value.detach().float().cpu())
                    for key, value in values.items()
                    if torch.is_tensor(value) and value.numel() == 1
                },
            }
            with (output / "events.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(event, sort_keys=True) + "\n")
            atomic_json(output / "progress.json", event)
            model._captured_vision.clear()
            if (
                cursor["global_step"] == 1
                or cursor["global_step"] % args.save_every == 0
            ):
                save_checkpoint(model, optimizer, output, cursor)
        cursor["epoch"] = epoch + 1
        boundary_checkpoint = save_checkpoint(
            model, optimizer, output, cursor
        )
        atomic_json(
            output / f"epoch{epoch + 1:02d}_evaluation_request.json",
            {
                "epoch": epoch + 1,
                "checkpoint": str(boundary_checkpoint),
                "status": "pending_official_dispatch",
            },
        )
    atomic_json(
        output / "status.json",
        {"status": "completed", **cursor},
    )


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except BaseException:
        output = Path(args.output_dir)
        if not args.preflight_only and output.is_dir():
            atomic_json(
                output / "status.json",
                {
                    "status": "failed",
                    "traceback": traceback.format_exc(),
                },
            )
        raise


if __name__ == "__main__":
    main()
