#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.collator_v1 import Stage1Collator
from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.diagnostics_v1 import teacher_forced_token_metrics
from human_lisa_qwen_v1.inference_v1 import load_stage1_checkpoint
from human_lisa_qwen_v1.schema_v1 import PROMPT_V1
from human_lisa_qwen_v1.training_v1 import move_to_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def single_token_ids(tokenizer: Any, texts: List[str]) -> List[int]:
    result = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            result.append(int(ids[0]))
    return sorted(set(result))


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if not torch.cuda.is_available():
        raise RuntimeError("teacher-forced audit requires CUDA")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda")
    model, processor, manifest = load_stage1_checkpoint(
        args.checkpoint,
        dtype=dtype,
        device=device,
    )
    processor.tokenizer.padding_side = "right"
    collator = Stage1Collator(
        processor=processor,
        token_ids=model.token_ids,
        prompt=PROMPT_V1,
        max_length=5000,
    )
    dataset = CocoUniHumanV1Dataset(
        args.annotation,
        args.image_root,
        training=False,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples,
        decode_eval_segmentation=True,
    )
    tracked = {
        "box": [model.token_ids["box"]],
        "pose": [model.token_ids["pose"]],
        "seg": [model.token_ids["seg"]],
        "comma": single_token_ids(processor.tokenizer, [","]),
        "close_array": single_token_ids(processor.tokenizer, ["]"]),
        "eos": [int(processor.tokenizer.eos_token_id)],
    }
    records: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        raw_batch = collator([sample])
        model_inputs = move_to_device(
            {
                name: raw_batch[name]
                for name in [
                    "input_ids",
                    "attention_mask",
                    "labels",
                    "pixel_values",
                    "image_grid_thw",
                ]
                if name in raw_batch
            },
            device,
        )
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=dtype, enabled=args.bf16
        ):
            outputs = model.base_model(
                **model_inputs,
                return_dict=True,
                use_cache=False,
                logits_to_keep=0,
            )
        metrics = teacher_forced_token_metrics(
            outputs.logits,
            model_inputs["labels"],
            tracked_token_ids=tracked,
        )
        record = {
            "image_id": int(sample["id"]),
            "persons": len(sample["instances"]),
            "ce_loss": float(outputs.loss.detach().float().cpu()),
            "target_tokens": raw_batch["label_audit"][0]["target_tokens"],
            "total_tokens": int(raw_batch["sequence_lengths"][0]),
            **metrics,
        }
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        del outputs, model_inputs, raw_batch

    supervised = sum(record["supervised_tokens"] for record in records)
    correct = sum(record["correct_tokens"] for record in records)
    summary = {
        "images": len(records),
        "persons": sum(record["persons"] for record in records),
        "mean_ce_loss": sum(record["ce_loss"] for record in records)
        / max(1, len(records)),
        "token_accuracy": correct / max(1, supervised),
        "exact_sequence_rate": sum(
            int(record["exact_sequence"]) for record in records
        )
        / max(1, len(records)),
        "records": records,
        "checkpoint": args.checkpoint,
        "manifest": manifest,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
