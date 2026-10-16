#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from pycocotools.coco import COCO

from human_lisa_qwen_v1.official_eval_v1 import _run_coco_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--inference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iou_xywh(first: Sequence[float], second: Sequence[float]) -> float:
    ax, ay, aw, ah = [float(v) for v in first]
    bx, by, bw, bh = [float(v) for v in second]
    ax2, ay2 = ax + max(0.0, aw), ay + max(0.0, ah)
    bx2, by2 = bx + max(0.0, bw), by + max(0.0, bh)
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / max(1e-12, union)


def score_sort(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted((dict(record) for record in records), key=lambda x: float(x.get("score", 0.0)), reverse=True)


def nms(records: Sequence[Dict[str, Any]], threshold: float) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for record in score_sort(records):
        if all(iou_xywh(record["bbox"], prior["bbox"]) < threshold for prior in kept):
            kept.append(record)
    return kept


def gt_by_image(coco_gt: COCO, image_ids: Sequence[int]) -> Dict[int, List[Dict[str, Any]]]:
    output: Dict[int, List[Dict[str, Any]]] = {}
    for image_id in image_ids:
        instances = []
        for ann in coco_gt.imgToAnns.get(int(image_id), []):
            bbox = ann.get("bbox")
            if int(ann.get("category_id", 1)) != 1:
                continue
            if int(ann.get("iscrowd", 0)) or int(ann.get("ignore", 0)):
                continue
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            if float(bbox[2]) <= 0.0 or float(bbox[3]) <= 0.0:
                continue
            instances.append(ann)
        output[int(image_id)] = instances
    return output


def greedy_matches(preds: Sequence[Dict[str, Any]], gts: Sequence[Dict[str, Any]], minimum_iou: float = 0.0) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            overlap = iou_xywh(pred["bbox"], gt["bbox"])
            if overlap >= minimum_iou:
                pairs.append((overlap, pi, gi))
    pairs.sort(reverse=True)
    used_pred, used_gt = set(), set()
    matches: List[Tuple[int, int, float]] = []
    for overlap, pi, gi in pairs:
        if pi in used_pred or gi in used_gt:
            continue
        used_pred.add(pi)
        used_gt.add(gi)
        matches.append((pi, gi, overlap))
    return matches


def variant_summary(by_image: Dict[int, List[Dict[str, Any]]], gt_map: Dict[int, List[Dict[str, Any]]]) -> Dict[str, Any]:
    images = len(gt_map)
    pred_total = sum(len(by_image.get(image_id, [])) for image_id in gt_map)
    gt_total = sum(len(gts) for gts in gt_map.values())
    matched = 0
    count_error = 0
    for image_id, gts in gt_map.items():
        preds = by_image.get(image_id, [])
        matched += len(greedy_matches(preds, gts, minimum_iou=0.5))
        count_error += abs(len(preds) - len(gts))
    return {
        "images": images,
        "gt_persons": gt_total,
        "pred_persons": pred_total,
        "person_count_mae": count_error / max(1, images),
        "matched_iou50": matched,
        "recall_iou50": matched / max(1, gt_total),
        "precision_iou50": matched / max(1, pred_total),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.inference))
    if args.max_samples is not None:
        rows = rows[: max(0, int(args.max_samples))]
    image_ids = [int(row["image_id"]) for row in rows]
    raw_by_image = {int(row["image_id"]): [dict(record) for record in row.get("records", [])] for row in rows}
    coco_gt = COCO(args.annotation)
    gt_map = gt_by_image(coco_gt, image_ids)

    variants: Dict[str, Dict[int, List[Dict[str, Any]]]] = {"original": raw_by_image}
    for threshold in (0.3, 0.5, 0.7):
        variants[f"score_ge_{threshold:.1f}"] = {
            image_id: [dict(record) for record in records if float(record.get("score", 0.0)) >= threshold]
            for image_id, records in raw_by_image.items()
        }
    for threshold in (0.5, 0.7, 0.8):
        variants[f"nms_{threshold:.1f}"] = {
            image_id: nms(records, threshold) for image_id, records in raw_by_image.items()
        }
    variants["nms_0.5_score_ge_0.3"] = {
        image_id: [record for record in nms(records, 0.5) if float(record.get("score", 0.0)) >= 0.3]
        for image_id, records in raw_by_image.items()
    }
    variants["oracle_gt_count_topscore"] = {
        image_id: score_sort(records)[: len(gt_map.get(image_id, []))]
        for image_id, records in raw_by_image.items()
    }
    variants["oracle_gt_count_nms_0.5"] = {
        image_id: nms(records, 0.5)[: len(gt_map.get(image_id, []))]
        for image_id, records in raw_by_image.items()
    }
    oracle_model_score: Dict[int, List[Dict[str, Any]]] = {}
    oracle_iou_score: Dict[int, List[Dict[str, Any]]] = {}
    for image_id, records in raw_by_image.items():
        matches = greedy_matches(records, gt_map.get(image_id, []), minimum_iou=0.0)
        selected_model, selected_iou = [], []
        for pred_index, _gt_index, overlap in matches:
            model_record = dict(records[pred_index])
            selected_model.append(model_record)
            iou_record = dict(model_record)
            iou_record["score"] = float(overlap)
            selected_iou.append(iou_record)
        oracle_model_score[image_id] = selected_model
        oracle_iou_score[image_id] = selected_iou
    variants["oracle_gt_iou_assignment_model_score"] = oracle_model_score
    variants["oracle_gt_iou_assignment_iou_score"] = oracle_iou_score

    results: Dict[str, Any] = {
        "protocol": "diagnostic_posthoc_bbox_only_val_gt_used_by_oracle_variants",
        "paper_eligibility": "diagnostic_only",
        "annotation": args.annotation,
        "inference": args.inference,
        "variants": {},
    }
    for name, by_image in variants.items():
        records = [record for image_id in image_ids for record in by_image.get(image_id, [])]
        print(json.dumps({"variant": name, "records": len(records)}), flush=True)
        bbox = _run_coco_eval(coco_gt, records, iou_type="bbox", image_ids=image_ids)
        results["variants"][name] = {
            **variant_summary(by_image, gt_map),
            **{f"bbox_{key}": value for key, value in bbox.items()},
        }

    original = results["variants"]["original"]
    expected_ap50 = 0.2323811028754709
    results["reproduction_check"] = {
        "expected_epoch1_bbox_AP50": expected_ap50,
        "observed_bbox_AP50": original["bbox_AP50"],
        "absolute_error": abs(original["bbox_AP50"] - expected_ap50),
        "ok": abs(original["bbox_AP50"] - expected_ap50) < 1e-9,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "reproduction_check": results["reproduction_check"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
