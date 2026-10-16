#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import Counter

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.schema import PROMPT, SPECIAL_TOKENS, build_target, load_jsonl, resolve_image_path
from human_lisa_qwen_v1.modeling import QwenLisaHeads, apply_lora_if_requested, load_qwen_vl_backbone


class V1LisaDataset(Dataset):
    def __init__(self, jsonl: str, image_root: str = "", max_samples: Optional[int] = None) -> None:
        self.rows = load_jsonl(jsonl)
        if max_samples is not None:
            self.rows = self.rows[: int(max_samples)]
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        image_path = resolve_image_path(row, self.image_root)
        image = Image.open(image_path).convert("RGB")
        target_text, aux = build_target(row)
        return {"id": row.get("id"), "image": image, "prompt": PROMPT, "target_text": target_text, "aux": aux}


class LisaCollator:
    def __init__(self, processor, tokenizer, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = int(max_length)

    def chat_prefix(self, prompt: str) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
        try:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prefixes = [self.chat_prefix(s["prompt"]) for s in samples]
        eos = self.tokenizer.eos_token or ""
        targets = [s["target_text"] + eos for s in samples]
        texts = [p + t for p, t in zip(prefixes, targets)]
        images = [s["image"] for s in samples]
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length)
        labels = batch["input_ids"].clone()
        prefix_lens = [len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prefixes]
        for i, plen in enumerate(prefix_lens):
            labels[i, : min(plen, labels.shape[1])] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        max_box = max((len(s["aux"]["box_targets"]) for s in samples), default=0)
        max_pose = max((len(s["aux"]["pose_targets"]) for s in samples), default=0)
        box_targets = torch.zeros((len(samples), max_box, 4), dtype=torch.float32)
        box_mask = torch.zeros((len(samples), max_box), dtype=torch.bool)
        pose_targets = torch.zeros((len(samples), max_pose, 17, 3), dtype=torch.float32)
        pose_mask = torch.zeros((len(samples), max_pose), dtype=torch.bool)
        for i, s in enumerate(samples):
            boxes = s["aux"]["box_targets"]
            poses = s["aux"]["pose_targets"]
            if boxes:
                box_targets[i, : len(boxes)] = torch.tensor(boxes, dtype=torch.float32)
                box_mask[i, : len(boxes)] = True
            if poses:
                pose_targets[i, : len(poses)] = torch.tensor(poses, dtype=torch.float32)
                pose_mask[i, : len(poses)] = True
        batch["box_targets"] = box_targets
        batch["box_target_mask"] = box_mask
        batch["pose_targets"] = pose_targets
        batch["pose_target_mask"] = pose_mask
        batch["ids"] = [s["id"] for s in samples]
        batch["target_texts"] = [s["target_text"] for s in samples]
        return batch


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif k not in ("ids", "target_texts"):
            out[k] = v
    return out


def dtype_from_args(args) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def hidden_size_from_config(model) -> int:
    cfg = getattr(model, "config", None)
    for obj in (cfg, getattr(cfg, "text_config", None), getattr(cfg, "llm_config", None)):
        if obj is not None and getattr(obj, "hidden_size", None):
            return int(getattr(obj, "hidden_size"))
    emb = model.get_input_embeddings()
    return int(emb.embedding_dim)


def lora_targets_from_preset(preset: str):
    preset = str(preset)
    llm_attn = ["q_proj", "k_proj", "v_proj", "o_proj"]
    visual_attn = ["attn.qkv", "attn.proj"]
    if preset == "llm_attn":
        return llm_attn
    if preset == "visual_attn":
        return visual_attn
    if preset == "llm_visual_attn":
        return llm_attn + visual_attn
    raise ValueError(f"unknown lora target preset: {preset}")


def summarize_trainable_params(model) -> Dict[str, Any]:
    counts = Counter()
    examples: Dict[str, List[str]] = {}
    total = 0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        n = int(param.numel())
        total += n
        if name.startswith("base_model.model.model.visual") or ".model.visual." in name or ".visual." in name:
            key = "visual"
        elif "lm_head" in name:
            key = "lm_head"
        elif "embed_tokens" in name:
            key = "embed_tokens"
        elif "lora_" in name:
            key = "language_or_other_lora"
        elif name.startswith("heads."):
            key = "token_heads"
        else:
            key = "other"
        counts[key] += n
        examples.setdefault(key, [])
        if len(examples[key]) < 5:
            examples[key].append(name)
    return {"trainable_total": total, "trainable_by_group": dict(counts), "trainable_examples": examples}


def add_special_tokens(processor, model) -> Dict[str, int]:
    tokenizer = processor.tokenizer
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    add = [t for t in SPECIAL_TOKENS if t not in existing]
    if add:
        tokenizer.add_special_tokens({"additional_special_tokens": existing + add})
        model.resize_token_embeddings(len(tokenizer))
    ids = {"box": tokenizer.convert_tokens_to_ids("<box>"), "pose": tokenizer.convert_tokens_to_ids("<pose>"), "seg": tokenizer.convert_tokens_to_ids("<seg>")}
    if any(v is None or int(v) < 0 for v in ids.values()):
        raise RuntimeError(f"failed to register special tokens: {ids}")
    return ids


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", default="/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/train.jsonl")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--model", default="/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--output-dir", default="/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/debug")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--freeze-base", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--lora", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-target-preset", choices=["llm_attn", "visual_attn", "llm_visual_attn"], default="llm_attn")
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--skip-save", action="store_true", help="Do not write checkpoints/final artifacts; intended only for batch-size probes.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ds = V1LisaDataset(args.train_jsonl, args.image_root, args.max_samples)

    if args.dry_run:
        sample = ds[0]
        print(json.dumps({"id": sample["id"], "target_text": sample["target_text"], "aux_shapes": {k: len(v) for k, v in sample["aux"].items()}}, ensure_ascii=False, indent=2))
        return

    dtype = dtype_from_args(args)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    base = load_qwen_vl_backbone(args.model, dtype)
    token_ids = add_special_tokens(processor, base)
    if args.freeze_base:
        for p in base.parameters():
            p.requires_grad = False
    lora_targets = lora_targets_from_preset(args.lora_target_preset)
    base = apply_lora_if_requested(base, args.lora, r=args.lora_r, alpha=args.lora_alpha, target_modules=lora_targets)
    wrapper = QwenLisaHeads(base, hidden_size_from_config(base), token_ids)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(device)
    wrapper.train()

    collator = LisaCollator(processor, processor.tokenizer, args.max_length)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collator)
    base_params, head_params = [], []
    for name, p in wrapper.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("heads."):
            head_params.append(p)
        else:
            base_params.append(p)
    optim = torch.optim.AdamW([
        {"params": base_params, "lr": args.lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=float(args.weight_decay))

    step = 0
    optim_step = 0
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    optim.zero_grad(set_to_none=True)
    steps_per_epoch = max(1, len(loader))
    if args.max_steps is not None and int(args.max_steps) > 0:
        total_steps = int(args.max_steps)
        schedule_mode = "max_steps"
    else:
        total_steps = max(1, int(math.ceil(float(args.epochs) * steps_per_epoch)))
        schedule_mode = "epochs"
    trainable_summary = summarize_trainable_params(wrapper)
    print(json.dumps({
        "training_plan": schedule_mode,
        "dataset_size": len(ds),
        "batch_size": args.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": args.batch_size * grad_accum_steps,
        "optimizer_steps_estimate": int(math.ceil(total_steps / float(grad_accum_steps))),
        "steps_per_epoch": steps_per_epoch,
        "epochs": float(args.epochs),
        "total_steps": total_steps,
        "max_length": args.max_length,
        "lora_target_preset": args.lora_target_preset,
        "lora_target_modules": lora_targets,
        "skip_save": bool(args.skip_save),
        **trainable_summary,
    }, ensure_ascii=False), flush=True)
    while step < total_steps:
        for raw_batch in loader:
            step += 1
            batch = move_batch(raw_batch, device)
            with torch.autocast(device_type="cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype in (torch.float16, torch.bfloat16))):
                outputs = wrapper(**batch)
                loss = outputs["loss"]
            (loss / float(grad_accum_steps)).backward()
            do_optim_step = (step % grad_accum_steps == 0) or (step >= total_steps)
            grad_norm = None
            if do_optim_step:
                grad_norm_t = torch.nn.utils.clip_grad_norm_([p for p in wrapper.parameters() if p.requires_grad], 1.0)
                grad_norm = float(grad_norm_t.detach().float().cpu()) if isinstance(grad_norm_t, torch.Tensor) else float(grad_norm_t)
                optim.step()
                optim.zero_grad(set_to_none=True)
                optim_step += 1
            if step % args.log_every == 0 or step == 1 or do_optim_step:
                log = {k: float(v.detach().float().cpu()) for k, v in outputs.items() if isinstance(v, torch.Tensor) and v.numel() == 1}
                extra = {"optimizer_step": optim_step, "did_optimizer_step": bool(do_optim_step)}
                if grad_norm is not None:
                    extra["grad_norm"] = grad_norm
                print(json.dumps({"step": step, **extra, **log}, ensure_ascii=False), flush=True)
            if (not args.skip_save) and (step % args.save_every == 0 or step == total_steps):
                ckpt = out / f"checkpoint-{step}"
                wrapper.save_heads(str(ckpt))
                processor.save_pretrained(ckpt / "processor")
            if step >= total_steps:
                break
    if not args.skip_save:
        wrapper.save_heads(str(out / "final"))
        processor.save_pretrained(out / "final" / "processor")


if __name__ == "__main__":
    main()
