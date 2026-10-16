#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pycocotools import mask as mask_utils

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset


def iou_xywh_xyxy(pred_xywh: List[float], gt_xyxy: List[float]) -> float:
    px, py, pw, ph = [float(v) for v in pred_xywh]
    ax1, ay1, ax2, ay2 = px, py, px + max(0.0, pw), py + max(0.0, ph)
    bx1, by1, bx2, by2 = [float(v) for v in gt_xyxy]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def greedy_match(preds: List[Dict[str, Any]], gts: List[Dict[str, Any]], thr: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for pi, pred in enumerate(preds):
        bbox = pred.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        for gi, gt in enumerate(gts):
            ov = iou_xywh_xyxy(bbox, gt["bbox_xyxy"])
            if ov >= thr:
                pairs.append((ov, pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for ov, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        out.append((pi, gi, ov))
    return out


def pose_pck(pred_kpts: Any, gt_kpts: np.ndarray, gt_box: List[float], thr: float = 0.2) -> Tuple[int, int]:
    if not isinstance(pred_kpts, list) or len(pred_kpts) < 51:
        return 0, 0
    try:
        pred = np.asarray(pred_kpts[:51], dtype=np.float32).reshape(17, 3)
    except Exception:
        return 0, 0
    scale = max(1.0, math.sqrt(max(0.0, float(gt_box[2]) - float(gt_box[0])) * max(0.0, float(gt_box[3]) - float(gt_box[1]))))
    ok = total = 0
    for p, g in zip(pred, gt_kpts[:17]):
        if float(g[2]) <= 0:
            continue
        total += 1
        dist = math.hypot(float(p[0]) - float(g[0]), float(p[1]) - float(g[1]))
        ok += int(dist <= thr * scale)
    return ok, total


def decode_pred_mask(seg: Any, height: int, width: int):
    if not seg:
        return None
    try:
        rle = seg
        if isinstance(rle, dict) and isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, height, width)
        mask = mask_utils.decode(rle)
        if mask.ndim == 3:
            mask = np.any(mask, axis=2)
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (height, width):
            return None
        return mask
    except Exception:
        return None


def mask_iou(pred_mask, gt_mask) -> float:
    if pred_mask is None or gt_mask is None:
        return float("nan")
    inter = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
    union = np.logical_or(pred_mask, gt_mask).sum(dtype=np.float64)
    return float(inter / max(1.0, union))


def load_inference(path: Path) -> List[Dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute(args) -> Dict[str, Any]:
    dataset = CocoUniHumanV1Dataset(args.annotation, args.image_root, training=False, horizontal_flip_prob=0.0, max_samples=None, decode_eval_segmentation=True)
    gt_by_image = {int(sample["id"]): sample for sample in dataset}
    infer_rows = load_inference(Path(args.inference))
    agg: Dict[str, Any] = {
        "images": 0, "schema_ok": 0, "health_ok": 0,
        "gt_persons": 0, "pred_persons": 0, "matched": 0,
        "pose_ok": 0, "pose_total": 0,
        "seg_valid_matched": 0, "seg_iou_sum": 0.0, "seg_iou50": 0,
        "gender_gt_matched": 0, "gender_answered": 0, "gender_correct": 0,
        "age_gt_matched": 0, "age_answered": 0, "age_abs_error_sum": 0.0,
        "count_abs_error_sum": 0,
    }
    for row in infer_rows:
        image_id = int(row.get("image_id", -1))
        sample = gt_by_image.get(image_id)
        if sample is None:
            continue
        gts = list(sample.get("instances", []))
        preds = list(row.get("records", []))
        matches = greedy_match(preds, gts, args.iou_thr)
        agg["images"] += 1
        agg["schema_ok"] += int(bool(row.get("schema_ok")))
        agg["health_ok"] += int(bool((row.get("health") or {}).get("ok")))
        agg["gt_persons"] += len(gts)
        agg["pred_persons"] += len(preds)
        agg["matched"] += len(matches)
        agg["count_abs_error_sum"] += abs(len(preds) - len(gts))
        height, width = int(sample["height"]), int(sample["width"])
        for pi, gi, _ov in matches:
            pred, gt = preds[pi], gts[gi]
            if bool(gt.get("pose_valid")):
                ok, total = pose_pck(pred.get("keypoints"), gt.get("keypoints"), gt.get("bbox_xyxy"))
                agg["pose_ok"] += ok
                agg["pose_total"] += total
            if bool(gt.get("seg_valid")) and gt.get("mask") is not None:
                miou = mask_iou(decode_pred_mask(pred.get("segmentation"), height, width), gt.get("mask"))
                if not math.isnan(miou):
                    agg["seg_valid_matched"] += 1
                    agg["seg_iou_sum"] += miou
                    agg["seg_iou50"] += int(miou >= 0.5)
            if bool(gt.get("gender_valid")):
                agg["gender_gt_matched"] += 1
                if "gender" in pred:
                    agg["gender_answered"] += 1
                    agg["gender_correct"] += int(int(pred.get("gender")) == int(gt.get("gender")))
            if bool(gt.get("age_valid")):
                agg["age_gt_matched"] += 1
                if "age" in pred:
                    agg["age_answered"] += 1
                    agg["age_abs_error_sum"] += abs(float(pred.get("age")) - float(gt.get("age")))
    metrics = dict(agg)
    metrics.update({
        "schema_ok_rate": agg["schema_ok"] / max(1, agg["images"]),
        "health_ok_rate": agg["health_ok"] / max(1, agg["images"]),
        "box_recall_iou50": agg["matched"] / max(1, agg["gt_persons"]),
        "box_precision_iou50": agg["matched"] / max(1, agg["pred_persons"]),
        "person_count_mae": agg["count_abs_error_sum"] / max(1, agg["images"]),
        "pose_pck@0.2_matched": agg["pose_ok"] / max(1, agg["pose_total"]),
        "seg_miou_matched": agg["seg_iou_sum"] / max(1, agg["seg_valid_matched"]),
        "seg_iou50_matched": agg["seg_iou50"] / max(1, agg["seg_valid_matched"]),
        "gender_coverage_on_matched_valid_gt": agg["gender_answered"] / max(1, agg["gender_gt_matched"]),
        "gender_acc_answered_only": agg["gender_correct"] / max(1, agg["gender_answered"]),
        "gender_acc_unknown_as_wrong": agg["gender_correct"] / max(1, agg["gender_gt_matched"]),
        "age_coverage_on_matched_valid_gt": agg["age_answered"] / max(1, agg["age_gt_matched"]),
        "age_mae_answered_only": agg["age_abs_error_sum"] / max(1, agg["age_answered"]),
        "protocol": "diagnostic_bbox_iou50_matched_from_same_inference_jsonl",
        "note": "Matched-only diagnostic from the same generated inference file, not official COCO AP/OKS/mask AP.",
        "inference": str(args.inference),
    })
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--inference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    args = parser.parse_args()
    total = len(CocoUniHumanV1Dataset(args.annotation, args.image_root, training=False, horizontal_flip_prob=0.0, decode_eval_segmentation=False))
    while True:
        rows = load_inference(Path(args.inference))
        print(json.dumps({"seen": len(rows), "total": total, "waiting": bool(args.wait and len(rows) < total)}, ensure_ascii=False), flush=True)
        if not args.wait or len(rows) >= total:
            break
        time.sleep(max(10, int(args.poll_seconds)))
    metrics = compute(args)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
