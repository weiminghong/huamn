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
from human_lisa_qwen_v1.inference_v1 import (
    assemble_coco_records,
    load_stage1_checkpoint,
)
from human_lisa_qwen_v1.official_eval_v1 import evaluate_all_official
from human_lisa_qwen_v1.schema_v1 import PROMPT_V1
from human_lisa_qwen_v1.training_v1 import move_to_device


DEFAULT_ATTR_EVAL = (
    REPO_ROOT
    / "HumanQueryNet_mmdet3_port"
    / "datasets"
    / "api_wrappers"
    / "eval_attr.py"
)


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


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if not torch.cuda.is_available():
        raise RuntimeError("oracle-record head audit requires CUDA")
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
    all_records: List[Dict[str, Any]] = []
    image_ids: List[int] = []
    per_image: List[Dict[str, Any]] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        raw_batch = collator([sample])
        model_batch = move_to_device(
            {
                name: raw_batch[name]
                for name in [
                    "input_ids",
                    "attention_mask",
                    "mm_token_type_ids",
                    "labels",
                    "pixel_values",
                    "image_grid_thw",
                    "box_targets",
                    "pose_targets",
                    "pose_areas",
                    "seg_targets",
                    "box_person_indices",
                    "pose_person_indices",
                    "seg_person_indices",
                    "original_sizes",
                ]
                if name in raw_batch
            },
            device,
        )
        with torch.no_grad(), torch.autocast(
            "cuda", dtype=dtype, enabled=args.bf16
        ):
            outputs = model(**model_batch, return_predictions=True)
        prediction = outputs["predictions"][0]
        obj = json.loads(raw_batch["target_texts"][0])
        persons = len(obj["persons"])
        records = assemble_coco_records(
            image_id=int(sample["id"]),
            width=int(sample["width"]),
            height=int(sample["height"]),
            obj=obj,
            boxes=prediction["boxes"],
            quality=prediction["box_quality"],
            token_probabilities=torch.ones(
                persons, device=device, dtype=torch.float32
            ),
            pose_xy=prediction["pose_xy"],
            pose_visibility_logits=prediction["pose_visibility"],
            mask_logits=prediction["mask_logits"],
        )
        all_records.extend(records)
        image_ids.append(int(sample["id"]))
        row = {
            "image_id": int(sample["id"]),
            "persons": persons,
            "box_l1": float(outputs["box_l1"].detach().float().cpu()),
            "pose_l1": float(outputs["pose_l1"].detach().float().cpu()),
            "pose_oks": float(outputs["pose_oks"].detach().float().cpu()),
            "pose_heatmap": float(outputs["pose_heatmap"].detach().float().cpu()),
            "mask_bce": float(outputs["mask_bce"].detach().float().cpu()),
            "mask_dice": float(outputs["mask_dice"].detach().float().cpu()),
        }
        per_image.append(row)
        print(json.dumps(row), flush=True)

    official = evaluate_all_official(
        args.annotation,
        all_records,
        image_ids=image_ids,
        attr_eval_source=DEFAULT_ATTR_EVAL,
    )
    metrics = {
        key: value
        for key, value in official.items()
        if key.startswith(("bbox_", "keypoints_", "segm_"))
    }
    result = {
        **metrics,
        "images": len(dataset),
        "persons": len(all_records),
        "per_image": per_image,
        "checkpoint": args.checkpoint,
        "protocol": "diagnostic_gt_person_records_task_heads",
        "attributes": "oracle_labels_not_reported",
        "manifest": manifest,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
