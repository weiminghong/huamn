import argparse
import importlib.util
import json
import math
import os
import os.path as osp
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from human.common.stage1_five_task_dataset import Stage1FiveTaskDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 five-task Qwen-VL training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_vl_stage1_five_tasks.py",
        help="stage1 five-task config file path",
    )
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--mask-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = osp.abspath(path)
    spec = importlib.util.spec_from_file_location("stage1_five_task_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"CONFIG not found in {cfg_path}")
    return dict(module.CONFIG)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_torch_dtype(cfg: Dict[str, Any]):
    if cfg.get("bf16", False):
        return torch.bfloat16
    if cfg.get("fp16", False):
        return torch.float16
    return torch.float32


def build_model(model_name_or_path: str, torch_dtype):
    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


def maybe_apply_lora(model, cfg: Dict[str, Any]):
    lora_cfg = cfg.get("lora", {})
    if not lora_cfg or not lora_cfg.get("enable", False):
        return model

    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.get("target_modules", [])),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


class FiveTaskCollator:
    def __init__(self, processor, tokenizer, max_length: int, seg_token_id: int) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.seg_token_id = seg_token_id

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompts = [s["prompt"] + "\n" for s in samples]
        targets = [s["target_text"] for s in samples]
        texts = [p + t for p, t in zip(prompts, targets)]
        images = [s["image"] for s in samples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        if "input_ids" not in batch:
            raise KeyError("Processor output has no input_ids, cannot build labels")

        labels = batch["input_ids"].clone()
        prefix_lens = [
            len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prompts
        ]
        for i, plen in enumerate(prefix_lens):
            labels[i, : min(plen, labels.shape[1])] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels

        seg_batch_indices: List[int] = []
        seg_token_indices: List[int] = []
        gt_masks: List[torch.Tensor] = []
        for i, sample in enumerate(samples):
            positions = (
                ((batch["input_ids"][i] == self.seg_token_id) & (labels[i] != -100))
                .nonzero(as_tuple=False)
                .flatten()
                .tolist()
            )
            masks = sample.get("mask_tensors", [])
            match_n = min(len(positions), len(masks))
            for j in range(match_n):
                seg_batch_indices.append(i)
                seg_token_indices.append(int(positions[j]))
                gt_masks.append(masks[j].float())

        if seg_batch_indices:
            batch["seg_batch_indices"] = torch.tensor(seg_batch_indices, dtype=torch.long)
            batch["seg_token_indices"] = torch.tensor(seg_token_indices, dtype=torch.long)
        else:
            batch["seg_batch_indices"] = torch.empty((0,), dtype=torch.long)
            batch["seg_token_indices"] = torch.empty((0,), dtype=torch.long)
        batch["gt_masks"] = gt_masks
        batch["sample_ids"] = [s["id"] for s in samples]
        return batch


class ParsingProjector(nn.Module):
    def __init__(self, text_dim: int, out_dim: int = 256) -> None:
        super().__init__()
        groups = 32 if out_dim % 32 == 0 else 1
        self.text_proj = nn.Linear(text_dim, out_dim)
        self.vis_proj = nn.Sequential(
            nn.LazyConv2d(out_dim, kernel_size=1),
            nn.GroupNorm(groups, out_dim),
            nn.ReLU(),
            nn.Conv2d(out_dim, out_dim, kernel_size=1),
        )

    def forward(self, text_embeddings: torch.Tensor, visual_feature_map: torch.Tensor) -> torch.Tensor:
        # text_embeddings: [N, text_dim]
        # visual_feature_map: [N, C, H, W]
        text_feat = self.text_proj(text_embeddings).unsqueeze(-1).unsqueeze(-1)  # [N, D, 1, 1]
        vis_feat = self.vis_proj(visual_feature_map)  # [N, D, H, W]
        mask_logits = torch.sum(text_feat * vis_feat, dim=1, keepdim=True)  # [N, 1, H, W]
        return mask_logits


def _seq_to_2d_feature_map(seq_feat: torch.Tensor) -> torch.Tensor:
    # seq_feat: [B, N, C]
    bsz, n_tok, channel = seq_feat.shape
    side = int(math.sqrt(n_tok))
    side = max(side, 1)
    usable = side * side
    if usable > n_tok:
        side = int(math.sqrt(max(n_tok - 1, 1)))
        side = max(side, 1)
        usable = side * side
    feat = seq_feat[:, :usable, :]
    return feat.transpose(1, 2).reshape(bsz, channel, side, side)


def extract_visual_feature_map(outputs, model_inputs: Dict[str, Any]) -> Optional[torch.Tensor]:
    for key in ("image_hidden_states", "vision_hidden_states"):
        value = getattr(outputs, key, None)
        if value is None:
            continue
        if isinstance(value, (list, tuple)) and value:
            value = value[-1]
        if torch.is_tensor(value):
            if value.ndim == 4:
                return value
            if value.ndim == 3:
                return _seq_to_2d_feature_map(value)

    hidden_states = getattr(outputs, "hidden_states", None)
    if isinstance(hidden_states, (list, tuple)) and hidden_states:
        last_hidden = hidden_states[-1]
        if torch.is_tensor(last_hidden) and last_hidden.ndim == 3:
            return _seq_to_2d_feature_map(last_hidden)

    pixel_values = model_inputs.get("pixel_values", None)
    if torch.is_tensor(pixel_values):
        if pixel_values.ndim == 5:
            pixel_values = pixel_values[:, 0]
        if pixel_values.ndim == 4:
            return pixel_values
    return None


def calc_mask_loss(mask_logits: torch.Tensor, gt_masks: List[torch.Tensor]) -> torch.Tensor:
    # mask_logits: [N, 1, H, W], gt_masks: list([H_i, W_i], float in {0,1})
    if mask_logits.shape[0] == 0:
        return mask_logits.new_tensor(0.0)

    losses: List[torch.Tensor] = []
    for i, gt_mask in enumerate(gt_masks):
        gt = gt_mask.to(mask_logits.device, dtype=mask_logits.dtype).unsqueeze(0).unsqueeze(0)
        pred = mask_logits[i : i + 1]  # [1,1,h,w]
        pred_up = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=False)

        bce = F.binary_cross_entropy_with_logits(pred_up, gt)
        pred_prob = torch.sigmoid(pred_up)
        intersection = (pred_prob * gt).sum(dim=(2, 3))
        union = pred_prob.sum(dim=(2, 3)) + gt.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        losses.append(bce + dice.mean())
    return torch.stack(losses).mean()


def build_optimizer(model, projector, cfg: Dict[str, Any]) -> AdamW:
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_params.extend([p for p in projector.parameters() if p.requires_grad])
    return AdamW(
        params=trainable_params,
        lr=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
    )


def move_model_inputs_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    model_inputs: Dict[str, Any] = {}
    for key, value in batch.items():
        if key in ("gt_masks", "sample_ids"):
            continue
        if torch.is_tensor(value):
            model_inputs[key] = value.to(device)
    return model_inputs


def forward_loss(
    model,
    projector,
    batch: Dict[str, Any],
    device: torch.device,
    mask_loss_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    model_inputs = move_model_inputs_to_device(batch, device)
    outputs = model(
        **model_inputs,
        output_hidden_states=True,
        return_dict=True,
    )
    text_loss = outputs.loss
    if text_loss is None:
        text_loss = model_inputs["input_ids"].new_tensor(0.0, dtype=torch.float32)

    mask_loss = text_loss.new_tensor(0.0)
    seg_batch_indices = model_inputs.get("seg_batch_indices", None)
    seg_token_indices = model_inputs.get("seg_token_indices", None)
    gt_masks = batch.get("gt_masks", [])

    has_seg = (
        seg_batch_indices is not None
        and seg_token_indices is not None
        and seg_batch_indices.numel() > 0
        and len(gt_masks) > 0
    )
    if has_seg:
        visual_map = extract_visual_feature_map(outputs=outputs, model_inputs=model_inputs)
        hidden_states = getattr(outputs, "hidden_states", None)
        if visual_map is not None and isinstance(hidden_states, (list, tuple)) and hidden_states:
            last_hidden = hidden_states[-1]
            token_emb = last_hidden[seg_batch_indices, seg_token_indices]  # [N, C]
            vis_sel = visual_map.index_select(0, seg_batch_indices)  # [N, C, H, W]
            mask_logits = projector(token_emb, vis_sel)
            mask_loss = calc_mask_loss(mask_logits=mask_logits, gt_masks=gt_masks)

    total_loss = text_loss + mask_loss_weight * mask_loss
    return total_loss, text_loss.detach(), mask_loss.detach()


def run_eval(
    model,
    projector,
    dataloader: DataLoader,
    device: torch.device,
    mask_loss_weight: float,
    max_steps: Optional[int] = None,
) -> Dict[str, float]:
    model.eval()
    projector.eval()

    total = 0
    sum_total_loss = 0.0
    sum_text_loss = 0.0
    sum_mask_loss = 0.0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if max_steps is not None and step >= max_steps:
                break
            total_loss, text_loss, mask_loss = forward_loss(
                model=model,
                projector=projector,
                batch=batch,
                device=device,
                mask_loss_weight=mask_loss_weight,
            )
            total += 1
            sum_total_loss += float(total_loss.item())
            sum_text_loss += float(text_loss.item())
            sum_mask_loss += float(mask_loss.item())

    model.train()
    projector.train()
    if total == 0:
        return {}
    return {
        "eval/total_loss": sum_total_loss / total,
        "eval/text_loss": sum_text_loss / total,
        "eval/mask_loss": sum_mask_loss / total,
    }


def save_checkpoint(
    model,
    processor,
    tokenizer,
    projector,
    output_dir: str,
    epoch: int,
    global_step: int,
    seg_token_id: int,
) -> None:
    ckpt_dir = osp.join(output_dir, f"checkpoint-epoch{epoch:02d}-step{global_step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    model.save_pretrained(ckpt_dir)
    processor.save_pretrained(osp.join(ckpt_dir, "processor"))
    tokenizer.save_pretrained(osp.join(ckpt_dir, "tokenizer"))
    torch.save(
        {
            "projector": projector.state_dict(),
            "seg_token_id": seg_token_id,
        },
        osp.join(ckpt_dir, "parsing_projector.pt"),
    )


def save_final(
    model,
    processor,
    tokenizer,
    projector,
    output_dir: str,
    seg_token_id: int,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(osp.join(output_dir, "processor"))
    tokenizer.save_pretrained(osp.join(output_dir, "tokenizer"))
    torch.save(
        {
            "projector": projector.state_dict(),
            "seg_token_id": seg_token_id,
        },
        osp.join(output_dir, "parsing_projector.pt"),
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.train_jsonl is not None:
        cfg["train_jsonl"] = args.train_jsonl
    if args.val_jsonl is not None:
        cfg["val_jsonl"] = args.val_jsonl
    if args.image_root is not None:
        cfg["image_root"] = args.image_root
    if args.mask_root is not None:
        cfg["mask_root"] = args.mask_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    os.makedirs(cfg["output_dir"], exist_ok=True)
    set_seed(int(cfg.get("seed", 42)))

    train_dataset = Stage1FiveTaskDataset(
        jsonl_path=cfg["train_jsonl"],
        image_root=cfg.get("image_root", None),
        mask_root=cfg.get("mask_root", None),
        for_training=True,
        max_samples=cfg.get("max_train_samples", None),
        extra_prompt=cfg.get("extra_prompt", None),
        seg_token=cfg.get("seg_token", "<SEG>"),
    )
    val_dataset = None
    if cfg.get("val_jsonl"):
        val_dataset = Stage1FiveTaskDataset(
            jsonl_path=cfg["val_jsonl"],
            image_root=cfg.get("image_root", None),
            mask_root=cfg.get("mask_root", None),
            for_training=False,
            max_samples=cfg.get("max_val_samples", None),
            extra_prompt=cfg.get("extra_prompt", None),
            seg_token=cfg.get("seg_token", "<SEG>"),
        )

    model_name_or_path = cfg["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    seg_token = cfg.get("seg_token", "<SEG>")
    tokenizer.add_special_tokens({"additional_special_tokens": [seg_token]})
    seg_token_id = tokenizer.convert_tokens_to_ids(seg_token)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer = tokenizer

    model = build_model(model_name_or_path, get_torch_dtype(cfg))
    model.resize_token_embeddings(len(tokenizer))
    model = maybe_apply_lora(model, cfg)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    hidden_dim = getattr(model.config, "hidden_size", None)
    if hidden_dim is None and hasattr(model.config, "text_config"):
        hidden_dim = getattr(model.config.text_config, "hidden_size", None)
    if hidden_dim is None:
        hidden_dim = 1024
    hidden_dim = int(hidden_dim)
    projector = ParsingProjector(
        text_dim=hidden_dim,
        out_dim=int(cfg.get("parsing_projector_dim", 256)),
    )

    collator = FiveTaskCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 4096)),
        seg_token_id=seg_token_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=int(cfg.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=int(cfg.get("dataloader_num_workers", 2)),
        collate_fn=collator,
    )
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(cfg.get("eval_batch_size", 1)),
            shuffle=False,
            num_workers=int(cfg.get("dataloader_num_workers", 2)),
            collate_fn=collator,
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    projector.to(device)

    optimizer = build_optimizer(model=model, projector=projector, cfg=cfg)
    grad_acc_steps = max(1, int(cfg.get("gradient_accumulation_steps", 1)))
    num_epochs = int(cfg.get("num_train_epochs", 1))
    total_optim_steps = max(1, math.ceil(len(train_loader) * num_epochs / grad_acc_steps))
    warmup_steps = int(total_optim_steps * float(cfg.get("warmup_ratio", 0.03)))
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optim_steps,
    )

    state_path = osp.join(cfg["output_dir"], "train_state.jsonl")
    global_step = 0
    model.train()
    projector.train()
    optimizer.zero_grad(set_to_none=True)

    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        if "projector" in ckpt:
            projector.load_state_dict(ckpt["projector"], strict=False)

    for epoch in range(1, num_epochs + 1):
        running_total_loss = 0.0
        running_text_loss = 0.0
        running_mask_loss = 0.0
        seen_steps = 0

        for step, batch in enumerate(train_loader, start=1):
            total_loss, text_loss, mask_loss = forward_loss(
                model=model,
                projector=projector,
                batch=batch,
                device=device,
                mask_loss_weight=float(cfg.get("mask_loss_weight", 1.0)),
            )
            (total_loss / grad_acc_steps).backward()
            running_total_loss += float(total_loss.item())
            running_text_loss += float(text_loss.item())
            running_mask_loss += float(mask_loss.item())
            seen_steps += 1

            should_step = (step % grad_acc_steps == 0) or (step == len(train_loader))
            if should_step:
                max_grad_norm = float(cfg.get("max_grad_norm", 1.0))
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % int(cfg.get("logging_steps", 10)) == 0:
                    log = {
                        "epoch": epoch,
                        "global_step": global_step,
                        "train/total_loss": running_total_loss / max(seen_steps, 1),
                        "train/text_loss": running_text_loss / max(seen_steps, 1),
                        "train/mask_loss": running_mask_loss / max(seen_steps, 1),
                        "lr": scheduler.get_last_lr()[0],
                    }
                    with open(state_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log, ensure_ascii=False) + "\n")
                    print(json.dumps(log, ensure_ascii=False))

        if bool(cfg.get("eval_each_epoch", True)) and val_loader is not None:
            eval_metrics = run_eval(
                model=model,
                projector=projector,
                dataloader=val_loader,
                device=device,
                mask_loss_weight=float(cfg.get("mask_loss_weight", 1.0)),
                max_steps=None,
            )
            if eval_metrics:
                eval_metrics["epoch"] = epoch
                eval_metrics["global_step"] = global_step
                with open(state_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(eval_metrics, ensure_ascii=False) + "\n")
                print(json.dumps(eval_metrics, ensure_ascii=False))

        if int(cfg.get("save_epochs", 1)) > 0 and epoch % int(cfg.get("save_epochs", 1)) == 0:
            save_checkpoint(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                projector=projector,
                output_dir=cfg["output_dir"],
                epoch=epoch,
                global_step=global_step,
                seg_token_id=seg_token_id,
            )

    save_final(
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        projector=projector,
        output_dir=cfg["output_dir"],
        seg_token_id=seg_token_id,
    )


if __name__ == "__main__":
    main()
