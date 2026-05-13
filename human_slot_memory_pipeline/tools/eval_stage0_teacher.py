from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

from human_slot_memory_pipeline.human_slot_memory.config import load_config
from human_slot_memory_pipeline.human_slot_memory.prompts import (
    DEFAULT_ASSISTANT_PREFIX_STAGE0,
    STAGE0_REPAIR_PROMPT,
    build_stage0_probe_prompt,
)
from human_slot_memory_pipeline.human_slot_memory.stage0_xml import (
    contains_think_tokens,
    prediction_needs_repair,
    truncate_to_first_answer,
)
from human_slot_memory_pipeline.human_slot_memory.stage1_data import convert_raw_annotations_to_stage1_records
from qwen4b_coco_unihuman_ft.eval_scheme2 import evaluate_records_scheme2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-0 / Stage-1 teacher evaluation.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/stage0_eval20.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--output-name", type=str, default="")
    return parser.parse_args()


def load_model(model_path: str, adapter_path: str = "", use_bf16: bool = True):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return processor, model


def build_inputs(processor: Any, prompt: str, assistant_prefix: str, image: Image.Image, max_length: int) -> Dict[str, Any]:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + assistant_prefix
    return processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )


def repair_once(
    processor: Any,
    model: Any,
    device: torch.device,
    raw_text: str,
    max_length: int,
    max_new_tokens: int,
    pad_token_id: int | None,
) -> str:
    repair_text = STAGE0_REPAIR_PROMPT + raw_text.strip()
    messages = [{"role": "user", "content": [{"type": "text", "text": repair_text}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + DEFAULT_ASSISTANT_PREFIX_STAGE0
    inputs = processor(
        text=[text],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    pred_suffix = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return DEFAULT_ASSISTANT_PREFIX_STAGE0 + pred_suffix


def run_one(
    processor: Any,
    model: Any,
    image_path: str,
    prompt: str,
    max_length: int,
    max_new_tokens: int,
    repair_max_new_tokens: int,
) -> Dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    assistant_prefix = DEFAULT_ASSISTANT_PREFIX_STAGE0
    inputs = build_inputs(
        processor=processor,
        prompt=prompt,
        assistant_prefix=assistant_prefix,
        image=image,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    tokenizer = getattr(processor, "tokenizer", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else eos_token_id

    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )
    input_len = inputs["input_ids"].shape[1]
    pred_suffix = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    raw_prediction_text = assistant_prefix + pred_suffix

    image_size = image.size
    need_repair, cleaned_prediction_text = prediction_needs_repair(raw_prediction_text, image_size=image_size)
    repaired_prediction_text = ""
    repaired = False
    if need_repair:
        repaired_prediction_text = truncate_to_first_answer(
            repair_once(
                processor=processor,
                model=model,
                device=device,
                raw_text=raw_prediction_text,
                max_length=max_length,
                max_new_tokens=repair_max_new_tokens,
                pad_token_id=pad_token_id,
            )
        )
        repaired_need, repaired_cleaned = prediction_needs_repair(repaired_prediction_text, image_size=image_size)
        cleaned_prediction_text = repaired_cleaned
        repaired = not repaired_need

    return {
        "raw_prediction_text": raw_prediction_text,
        "prediction_text": truncate_to_first_answer(cleaned_prediction_text),
        "contains_raw_think": contains_think_tokens(raw_prediction_text),
        "contains_final_think": contains_think_tokens(cleaned_prediction_text),
        "repaired": repaired,
        "repair_attempted": need_repair,
        "repaired_prediction_text": repaired_prediction_text,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.ensure_output_dir()
    num_samples = int(args.num_samples or cfg.runtime.get("num_samples", 20))
    adapter_path = args.adapter_path or str(cfg.paths.get("adapter_path", ""))
    output_name = args.output_name or ("stage1_eval20" if adapter_path else "stage0_eval20")

    gt_records, source_stats = convert_raw_annotations_to_stage1_records(
        annotation_path=cfg.paths["val_ann_json"],
        image_root=cfg.paths["val_image_root"],
        max_samples=None,
        skip_missing_images=True,
    )
    indices = list(range(len(gt_records)))
    random.Random(int(cfg.project.get("seed", 42))).shuffle(indices)
    indices = indices[:num_samples]
    gt_subset = [gt_records[index] for index in indices]

    processor, model = load_model(
        str(cfg.paths["qwen_model"]),
        adapter_path=adapter_path,
        use_bf16=bool(cfg.optimization.get("bf16", True)),
    )
    model.eval().to(str(cfg.runtime.get("device", "cuda")))

    prompt = build_stage0_probe_prompt()
    pred_rows: List[Dict[str, Any]] = []
    raw_think_count = 0
    final_think_count = 0
    repaired_count = 0
    repair_attempt_count = 0
    for gt_record in gt_subset:
        result = run_one(
            processor=processor,
            model=model,
            image_path=str(gt_record["image_abs"]),
            prompt=prompt,
            max_length=int(cfg.stage1.get("max_length", 2048)),
            max_new_tokens=int(cfg.stage1.get("max_new_tokens", 768)),
            repair_max_new_tokens=int(cfg.stage1.get("repair_max_new_tokens", 512)),
        )
        raw_think_count += int(result["contains_raw_think"])
        final_think_count += int(result["contains_final_think"])
        repaired_count += int(result["repaired"])
        repair_attempt_count += int(result["repair_attempted"])
        pred_rows.append(
            {
                "id": gt_record["id"],
                "image": gt_record["image"],
                "image_abs": gt_record["image_abs"],
                "width": gt_record["width"],
                "height": gt_record["height"],
                "prediction_text": result["prediction_text"],
                "raw_prediction_text": result["raw_prediction_text"],
                "repair_attempted": result["repair_attempted"],
                "repaired": result["repaired"],
            }
        )

    metrics = evaluate_records_scheme2(
        gt_records=gt_subset,
        pred_records=pred_rows,
        iou_thr=float(cfg.runtime.get("iou_thr", 0.5)),
        pck_thr=float(cfg.runtime.get("pck_thr", 0.2)),
    )
    metrics["source_stats"] = source_stats
    metrics["adapter_path"] = adapter_path
    metrics["raw_think_rate"] = raw_think_count / max(num_samples, 1)
    metrics["final_think_rate"] = final_think_count / max(num_samples, 1)
    metrics["repair_attempt_rate"] = repair_attempt_count / max(num_samples, 1)
    metrics["repair_success_rate"] = repaired_count / max(repair_attempt_count, 1)

    pred_path = Path(output_dir) / f"{output_name}_predictions.jsonl"
    metrics_path = Path(output_dir) / f"{output_name}_metrics.json"
    with pred_path.open("w", encoding="utf-8") as handle:
        for row in pred_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)

    print(json.dumps({key: value for key, value in metrics.items() if key != "per_image"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
