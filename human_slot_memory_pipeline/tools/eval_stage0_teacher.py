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
    DEFAULT_ASSISTANT_PREFIX_STAGE1,
    build_stage1_probe_prompt,
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


def build_chat_text(processor: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + DEFAULT_ASSISTANT_PREFIX_STAGE1


def run_one(processor: Any, model: Any, image_path: str, prompt: str, max_length: int, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    text = build_chat_text(processor, prompt)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    input_len = inputs["input_ids"].shape[1]
    pred_suffix = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return DEFAULT_ASSISTANT_PREFIX_STAGE1 + pred_suffix


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

    prompt = build_stage1_probe_prompt()
    pred_rows: List[Dict[str, Any]] = []
    for gt_record in gt_subset:
        prediction_text = run_one(
            processor=processor,
            model=model,
            image_path=str(gt_record["image_abs"]),
            prompt=prompt,
            max_length=int(cfg.stage1.get("max_length", 2048)),
            max_new_tokens=int(cfg.stage1.get("max_new_tokens", 1024)),
        )
        pred_rows.append(
            {
                "id": gt_record["id"],
                "image": gt_record["image"],
                "image_abs": gt_record["image_abs"],
                "width": gt_record["width"],
                "height": gt_record["height"],
                "prediction_text": prediction_text,
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
