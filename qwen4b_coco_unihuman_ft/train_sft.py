import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

from qwen4b_coco_unihuman_ft.dataset import CocoUniHumanSFTDataset, SFTCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT training for COCO-UniHuman on Qwen models.")
    parser.add_argument(
        "--config",
        type=str,
        default="qwen4b_coco_unihuman_ft/configs/qwen4b_box_pose_sft.yaml",
    )
    return parser.parse_args()


def load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required. Install: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml config: {path}")
    return cfg


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dtype(cfg: Dict[str, Any]):
    if bool(cfg.get("bf16", False)):
        return torch.bfloat16
    if bool(cfg.get("fp16", False)):
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
    if not isinstance(lora_cfg, dict) or not bool(lora_cfg.get("enable", False)):
        return model

    from peft import LoraConfig, get_peft_model

    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)

    os.makedirs(cfg["output_dir"], exist_ok=True)
    with open(os.path.join(cfg["output_dir"], "resolved_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    set_seed(int(cfg.get("seed", 42)))

    model_name_or_path = str(cfg["model_name_or_path"])
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    train_dataset = CocoUniHumanSFTDataset(
        jsonl_path=str(cfg["train_jsonl"]),
        image_root=cfg.get("image_root", None),
        max_samples=cfg.get("max_train_samples", None),
        extra_prompt=cfg.get("extra_prompt", None),
    )
    val_dataset = None
    if cfg.get("val_jsonl"):
        val_dataset = CocoUniHumanSFTDataset(
            jsonl_path=str(cfg["val_jsonl"]),
            image_root=cfg.get("image_root", None),
            max_samples=cfg.get("max_val_samples", None),
            extra_prompt=cfg.get("extra_prompt", None),
        )

    model = build_model(model_name_or_path=model_name_or_path, torch_dtype=get_dtype(cfg))
    model = maybe_apply_lora(model, cfg)
    if bool(cfg.get("gradient_checkpointing", False)):
        model.gradient_checkpointing_enable()

    collator = SFTCollator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 4096)),
    )

    report_to = cfg.get("report_to", "none")
    if report_to in ("none", None):
        report_to = []

    has_eval = val_dataset is not None
    train_args = TrainingArguments(
        output_dir=str(cfg["output_dir"]),
        per_device_train_batch_size=int(cfg.get("train_batch_size", 1)),
        per_device_eval_batch_size=int(cfg.get("eval_batch_size", 1)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 1)),
        learning_rate=float(cfg.get("learning_rate", 1e-5)),
        weight_decay=float(cfg.get("weight_decay", 0.01)),
        num_train_epochs=float(cfg.get("num_train_epochs", 1)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        save_steps=int(cfg.get("save_steps", 500)),
        eval_steps=int(cfg.get("eval_steps", 500)),
        save_total_limit=int(cfg.get("save_total_limit", 2)),
        bf16=bool(cfg.get("bf16", False)),
        fp16=bool(cfg.get("fp16", False)),
        dataloader_num_workers=int(cfg.get("dataloader_num_workers", 2)),
        remove_unused_columns=False,
        report_to=report_to,
        evaluation_strategy="steps" if has_eval else "no",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model()
    if has_eval:
        metrics = trainer.evaluate()
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))


if __name__ == "__main__":
    main()
