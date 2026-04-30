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
from qwen4b_coco_unihuman_ft.prompts import (
    DEFAULT_ASSISTANT_PREFIX_TAG,
    build_vl_chat_text,
)


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


def _jsonable_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in cfg.items():
        try:
            json.dumps(value)
            out[str(key)] = value
        except Exception:
            out[str(key)] = str(value)
    return out


def _prepare_preview_inputs(processor, prompt: str, image, max_length: int) -> Dict[str, torch.Tensor]:
    batch = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )
    return batch


def _move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in batch.items():
        out[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return out


def _filter_generate_inputs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    accepted = {
        "input_ids",
        "attention_mask",
        "position_ids",
        "token_type_ids",
        "past_key_values",
        "use_cache",
        "pixel_values",
        "image_grid_thw",
        "images",
    }
    return {k: v for k, v in inputs.items() if k in accepted}


def _log_swanlab_predictions(
    trainer: Trainer,
    processor,
    dataset: CocoUniHumanSFTDataset,
    cfg: Dict[str, Any],
) -> None:
    swanlab_cfg = cfg.get("swanlab", {})
    if not isinstance(swanlab_cfg, dict) or not bool(swanlab_cfg.get("enable", False)):
        return

    try:
        import swanlab
    except Exception as e:
        raise RuntimeError("SwanLab preview logging requested, but `swanlab` is not installed.") from e

    preview_count = min(int(swanlab_cfg.get("preview_samples", 4)), len(dataset))
    if preview_count <= 0:
        return

    model = trainer.model
    model.eval()
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else eos_token_id
    device = next(model.parameters()).device
    images_to_log = []

    with torch.no_grad():
        for idx in range(preview_count):
            sample = dataset[idx]
            prompt = build_vl_chat_text(
                processor=processor,
                prompt=str(sample["prompt"]),
                assistant_prefix=DEFAULT_ASSISTANT_PREFIX_TAG,
            )
            image = sample["image"]
            inputs = _prepare_preview_inputs(
                processor=processor,
                prompt=prompt,
                image=image,
                max_length=int(cfg.get("max_length", 4096)),
            )
            inputs = _move_batch_to_device(inputs, device)
            gen_inputs = _filter_generate_inputs(model, inputs)
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=int(swanlab_cfg.get("preview_max_new_tokens", 1024)),
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
            )
            input_len = gen_inputs["input_ids"].shape[1]
            pred_text = processor.batch_decode(gen_ids[:, input_len:], skip_special_tokens=True)[0]
            caption = (
                f"id={sample['id']}\n"
                f"prompt={sample['prompt']}\n\n"
                f"gt={sample['target_text']}\n\n"
                f"pred={DEFAULT_ASSISTANT_PREFIX_TAG}{pred_text}"
            )
            images_to_log.append(swanlab.Image(image, caption=caption))

    if images_to_log:
        swanlab.log({"Prediction": images_to_log})


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

    callbacks = []
    swanlab_cfg = cfg.get("swanlab", {})
    if isinstance(swanlab_cfg, dict) and bool(swanlab_cfg.get("enable", False)):
        try:
            from swanlab.integration.transformers import SwanLabCallback
        except Exception as e:
            raise RuntimeError(
                "SwanLab is enabled in config, but the Transformers integration is unavailable. "
                "Install it with: pip install swanlab"
            ) from e

        callbacks.append(
            SwanLabCallback(
                project=str(swanlab_cfg.get("project", "human-box-pose-attr-sft")),
                experiment_name=str(
                    swanlab_cfg.get(
                        "experiment_name",
                        os.path.basename(os.path.abspath(str(cfg["output_dir"]))),
                    )
                ),
                config=_jsonable_cfg(cfg),
            )
        )

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
        callbacks=callbacks,
    )
    trainer.train()
    trainer.save_model()
    if has_eval:
        metrics = trainer.evaluate()
        print(json.dumps(metrics, ensure_ascii=False, indent=2, default=float))
    preview_dataset = val_dataset if val_dataset is not None else train_dataset
    _log_swanlab_predictions(
        trainer=trainer,
        processor=processor,
        dataset=preview_dataset,
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
