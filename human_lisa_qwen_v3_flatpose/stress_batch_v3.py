#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoProcessor

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v2.collator_v2 import Stage1V2Collator
from human_lisa_qwen_v2.schema_v2 import PROMPT_V2
from human_lisa_qwen_v3_flatpose.train_stage1_v3 import (
    MODEL_INPUT_KEYS,
    audit_lora_coverage,
    build_model,
)
from human_lisa_qwen_v3_flatpose.training_v3 import (
    assert_finite_gradients,
    assert_finite_optimizer_state,
    build_optimizer,
    clip_global_gradient,
    gradient_statistics,
    move_to_device,
    scalar_tensors,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--model", default="/root/autodl-tmp/model/Qwen3.5-4B")
    parser.add_argument("--annotation", default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json")
    parser.add_argument("--image-root", default="/root/autodl-tmp/coco/train2017")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    dtype = torch.bfloat16
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    model_args = SimpleNamespace(
        model=args.model, lora_r=16, lora_alpha=32, lora_dropout=0.05,
        lora_target_mode="all-linear", gradient_checkpointing=True,
        vision_layers=[8, 16, 24], decoder_dim=256, decoder_heads=8,
        person_query_layers=3, pose_decoder_layers=2, max_people=25,
        query_objectness_weight=1.0, query_bbox_l1_weight=5.0,
        query_giou_weight=2.0, query_aux_weight=0.5,
        pose_l1_weight=10.0, pose_oks_weight=4.0,
        pose_visibility_weight=1.0, pose_aux_weight=0.5,
        mask_bce_weight=2.0, mask_dice_weight=5.0,
    )
    model = build_model(model_args, processor, dtype)
    coverage = audit_lora_coverage(model, "all-linear")
    device = torch.device("cuda")
    model.to(device).train()
    dataset = CocoUniHumanV1Dataset(
        args.annotation, args.image_root, training=True, horizontal_flip_prob=0.0,
    )
    if len(set(args.indices)) != len(args.indices):
        raise ValueError("stress indices must be unique")
    samples = [dataset[index] for index in args.indices]
    collator = Stage1V2Collator(
        processor=processor, token_ids=model.token_ids, prompt=PROMPT_V2, max_length=5000,
    )
    raw_batch = collator(samples)
    batch = move_to_device(
        {name: value for name, value in raw_batch.items() if name in MODEL_INPUT_KEYS}, device
    )
    optimizer, optimizer_manifest = build_optimizer(
        model, visual_lr=1e-5, language_lr=1e-5, token_lr=2e-5,
        head_lr=2e-4, weight_decay=0.01,
    )
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.autocast("cuda", dtype=dtype):
        result = model(**batch)
    if not torch.isfinite(result["loss"]):
        raise FloatingPointError("non-finite stress loss")
    result["loss"].backward()
    assert_finite_gradients(model)
    gradients = gradient_statistics(model)
    if any(not item["finite"] or item["nonzero"] <= 0 for item in gradients.values()):
        raise RuntimeError(f"stress gradient gate failed: {gradients}")
    preclip = clip_global_gradient(model, 1.0)
    optimizer.step()
    assert_finite_optimizer_state(optimizer, model)
    payload = {
        "protocol": "stage1_v3_alllinear_worst_case_batch_forward_backward_adamw",
        "indices": args.indices,
        "image_ids": raw_batch["sample_ids"],
        "persons_per_image": raw_batch["num_persons"],
        "sequence_lengths": raw_batch["sequence_lengths"].tolist(),
        "max_sequence_tokens": int(raw_batch["sequence_lengths"].max()),
        "lora_coverage": coverage,
        "optimizer": optimizer_manifest,
        "preclip_global_grad_norm": preclip,
        "gradient_families": gradients,
        "losses": scalar_tensors(result),
        "cuda_allocated_mb": round(torch.cuda.memory_allocated(device) / 1048576, 1),
        "cuda_reserved_mb": round(torch.cuda.memory_reserved(device) / 1048576, 1),
        "cuda_max_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1048576, 1),
        "cuda_max_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1048576, 1),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
