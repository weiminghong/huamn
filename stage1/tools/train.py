import argparse
import importlib.util
import json
import os
import os.path as osp
import random
import sys
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    TrainerCallback,
    Trainer,
    TrainingArguments,
)

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from human.common import (
    Stage1HumanDataset,
    evaluate_stage1_records,
    parse_json_object_from_text,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 Qwen-VL training")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_vl_stage1.py",
        help="stage1 config file path",
    )
    parser.add_argument("--train-jsonl", type=str, default=None)
    parser.add_argument("--val-jsonl", type=str, default=None)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = osp.abspath(path)
    spec = importlib.util.spec_from_file_location("stage1_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"CONFIG not found in {cfg_path}")
    cfg = dict(module.CONFIG)
    return cfg


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


class Stage1Collator:
    def __init__(self, processor, tokenizer, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
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
        return batch


def _collate_identity(batch):
    return batch


def run_generation_eval(
    model,
    processor,
    dataset,
    cfg: Dict[str, Any],
    max_samples: Optional[int] = 256,
) -> Dict[str, float]:
    if dataset is None or len(dataset) == 0:
        return {}

    if max_samples is None:
        max_samples = len(dataset)
    max_samples = max(1, min(int(max_samples), len(dataset)))
    subset = torch.utils.data.Subset(dataset, list(range(max_samples)))
    dataloader = DataLoader(
        subset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=0,
        collate_fn=_collate_identity,
    )

    device = next(model.parameters()).device
    max_length = int(cfg.get("max_length", 4096))
    max_new_tokens = int(cfg.get("eval_max_new_tokens", cfg.get("max_new_tokens", 512)))
    do_sample = not bool(cfg.get("eval_use_greedy", True))

    eval_records = []
    was_training = model.training
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            prompts = [s["prompt"] + "\n" for s in batch]
            images = [s["image"] for s in batch]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                num_beams=1,
            )
            input_len = inputs["input_ids"].shape[1]
            gen_suffix = gen_ids[:, input_len:]
            pred_texts = processor.batch_decode(gen_suffix, skip_special_tokens=True)

            for sample, pred_text in zip(batch, pred_texts):
                pred_struct = parse_json_object_from_text(pred_text)
                valid_json = pred_struct is not None
                if pred_struct is None:
                    pred_struct = {"box": [], "pose": [], "parsing": []}
                gt_struct = parse_json_object_from_text(sample["target_text"])
                if gt_struct is None:
                    gt_struct = {"box": [], "pose": [], "parsing": []}
                eval_records.append(
                    {
                        "id": sample["id"],
                        "valid_json": valid_json,
                        "pred_struct": pred_struct,
                        "gt_struct": gt_struct,
                    }
                )
    if was_training:
        model.train()
    return evaluate_stage1_records(eval_records)


class Stage1EpochEvalCallback(TrainerCallback):
    def __init__(self, cfg: Dict[str, Any], processor, eval_dataset):
        self.cfg = cfg
        self.processor = processor
        self.eval_dataset = eval_dataset
        self.history_path = osp.join(cfg["output_dir"], "stage1_epoch_eval_metrics.jsonl")
        self.trainer = None

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.eval_dataset is None:
            return control
        if not bool(self.cfg.get("eval_each_epoch", False)):
            return control

        metrics = run_generation_eval(
            model=model,
            processor=self.processor,
            dataset=self.eval_dataset,
            cfg=self.cfg,
            max_samples=self.cfg.get("eval_max_samples_each_epoch", 256),
        )
        if not metrics:
            return control

        metrics = {f"stage1/{k}": v for k, v in metrics.items()}
        metrics["stage1/epoch"] = float(state.epoch or 0.0)

        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

        trainer = self.trainer
        if trainer is not None:
            trainer.log(metrics)
        else:
            print(json.dumps(metrics, ensure_ascii=False))

        return control


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.train_jsonl is not None:
        cfg["train_jsonl"] = args.train_jsonl
    if args.val_jsonl is not None:
        cfg["val_jsonl"] = args.val_jsonl
    if args.image_root is not None:
        cfg["image_root"] = args.image_root
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    set_seed(int(cfg.get("seed", 42)))

    train_dataset = Stage1HumanDataset(
        jsonl_path=cfg["train_jsonl"],
        image_root=cfg.get("image_root", None),
        for_training=True,
        max_samples=cfg.get("max_train_samples", None),
        extra_prompt=cfg.get("extra_prompt", None),
    )

    eval_dataset = None
    if cfg.get("val_jsonl"):
        eval_dataset = Stage1HumanDataset(
            jsonl_path=cfg["val_jsonl"],
            image_root=cfg.get("image_root", None),
            for_training=False,
            max_samples=cfg.get("max_val_samples", None),
            extra_prompt=cfg.get("extra_prompt", None),
        )

    model_name_or_path = cfg["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)

    model = build_model(model_name_or_path, get_torch_dtype(cfg))
    model = maybe_apply_lora(model, cfg)
    if cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()

    collator = Stage1Collator(
        processor=processor,
        tokenizer=tokenizer,
        max_length=int(cfg.get("max_length", 4096)),
    )

    report_to = cfg.get("report_to", "none")
    if report_to in ("none", None):
        report_to = []

    has_eval = eval_dataset is not None
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
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
        load_best_model_at_end=has_eval,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    if eval_dataset is not None and bool(cfg.get("eval_each_epoch", False)):
        cb = Stage1EpochEvalCallback(cfg=cfg, processor=processor, eval_dataset=eval_dataset)
        cb.trainer = trainer
        trainer.add_callback(cb)

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(cfg["output_dir"])
    processor.save_pretrained(osp.join(cfg["output_dir"], "processor"))


if __name__ == "__main__":
    main()
