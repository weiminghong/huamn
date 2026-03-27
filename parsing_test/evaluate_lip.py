import argparse
import json
import os
import os.path as osp
import sys
from typing import Any, Dict, List

import numpy as np
from tqdm import tqdm

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from parsing_test.io_utils import dump_jsonl
from parsing_test.labels import LIP_LABELS
from parsing_test.lip_dataset import LipParsingDataset
from parsing_test.metrics import compute_confusion_matrix, summarize_iou
from parsing_test.qwen_sam2_pipeline import QwenSam2ParsingPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen-VL + SAM2 parsing baseline on LIP-style data.")
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--ann-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--qwen-model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--sam2-model", type=str, default="facebook/sam2.1-hiera-large")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--qwen-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--sam2-dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"])
    parser.add_argument("--save-vis", action="store_true")
    parser.add_argument("--extra-prompt", type=str, default=None)
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    pred_mask_dir = osp.join(args.output_dir, "pred_masks")
    ensure_dir(pred_mask_dir)

    dataset = LipParsingDataset(
        image_dir=args.image_dir,
        ann_dir=args.ann_dir,
        max_samples=args.max_samples,
    )
    if len(dataset) == 0:
        raise RuntimeError("No images found for evaluation.")

    pipeline = QwenSam2ParsingPipeline(
        qwen_model_name_or_path=args.qwen_model,
        sam2_model_name_or_path=args.sam2_model,
        max_new_tokens=args.max_new_tokens,
        qwen_dtype=args.qwen_dtype,
        sam2_dtype=args.sam2_dtype,
        device=args.device,
        extra_prompt=args.extra_prompt,
    )

    num_classes = len(LIP_LABELS)
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    results: List[Dict[str, Any]] = []

    for sample in tqdm(dataset, total=len(dataset), desc="evaluating"):
        pred = pipeline.predict(sample["image"])
        gt_mask = sample["gt_mask"]
        if gt_mask is None:
            raise RuntimeError(
                "Ground-truth masks were not found. Local mIoU evaluation requires annotation masks."
            )

        pred_mask = pred["semantic_map"]
        hist += compute_confusion_matrix(pred_mask, gt_mask, num_classes=num_classes)

        pred_mask_path = osp.join(pred_mask_dir, f'{sample["id"]}.png')
        pipeline.semantic_map_to_png(pred_mask, pred_mask_path)

        record = {
            "id": sample["id"],
            "image_path": sample["image_path"],
            "prediction_text": pred["prediction_text"],
            "prediction_struct": pred["prediction_struct"],
            "valid_json": pred["valid_json"],
            "predicted_parts": pred["predicted_parts"],
            "pred_mask_path": pred_mask_path,
        }
        results.append(record)

    metrics = summarize_iou(hist, LIP_LABELS)
    metrics["num_samples"] = len(dataset)

    dump_jsonl(osp.join(args.output_dir, "predictions.jsonl"), results)
    with open(osp.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
