from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from mmengine.config import Config
from mmdet.registry import MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_PORT = REPO_ROOT / "HumanQueryNet_mmdet3_port"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(HQNET_PORT) not in sys.path:
    sys.path.insert(0, str(HQNET_PORT))

import models  # noqa: F401,E402
import core  # noqa: F401,E402
from qwen_dinov3_hqnet_pipeline.tools.train_stage2_qwen_dino_hqnet_native import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_DINO,
    DEFAULT_QWEN,
    build_batch,
    collate_batches,
    load_index,
    make_cfg,
)
from qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone import FrozenQwenDINOv3Backbone  # noqa: F401,E402
from human_lisa_qwen_v1.official_eval_v1 import (  # noqa: E402
    encode_binary_mask,
    evaluate_all_official,
)

DEFAULT_VAL_ANN = "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
DEFAULT_VAL_ROOT = "/share2/program/weiminghong/coco/val2017"
DEFAULT_STAGE1_FINAL_ADAPTER = "/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4/train/final/base_or_adapter"
DEFAULT_STAGE2_CKPT = "/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260710_stage1final_qwen25_visual_lora_bs4_e1/latest.pt"
DEFAULT_OUTPUT_DIR = "/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260710_stage1final_qwen25_visual_lora_bs4_e1/eval_val_native_latest_gpu2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Native Stage2 val evaluation for Qwen+DINOv3+HumanQueryNet checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--backbone-mode", choices=["qwen_fusion", "dino_only"], default="qwen_fusion")
    parser.add_argument("--val-ann-file", default=DEFAULT_VAL_ANN)
    parser.add_argument("--val-image-root", default=DEFAULT_VAL_ROOT)
    parser.add_argument("--qwen-model", default="/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--qwen-adapter", default=DEFAULT_STAGE1_FINAL_ADAPTER)
    parser.add_argument("--dino-model", default=DEFAULT_DINO)
    parser.add_argument("--checkpoint", default=DEFAULT_STAGE2_CKPT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--profile", choices=["full", "lite"], default="full")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--num-query", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1, help="Keep 1: HumanQueryNet.simple_test returns only the first item for this port.")
    parser.add_argument("--max-instances", type=int, default=0)
    parser.add_argument("--limit-images", type=int, default=0)
    parser.add_argument("--score-thr", type=float, default=0.3, help="Threshold only for diagnostic matched predictions written per image.")
    parser.add_argument("--official-score-thr", type=float, default=0.0, help="Threshold for COCOeval/AttrCOCOeval export. Keep 0.0 for AP ranking.")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--fusion-mode", default="feature_add")
    parser.add_argument("--freeze-dino", action="store_true")
    parser.add_argument("--cuda-visible-devices", default="2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--attr-eval-source", default=str(REPO_ROOT / "HumanQueryNet_mmdet3_port" / "datasets" / "api_wrappers" / "eval_attr.py"))
    return parser.parse_args()


def iou_xyxy(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a[:4]]
    bx1, by1, bx2, by2 = [float(x) for x in b[:4]]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def match_boxes(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]], thr: float) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[float, int, int]] = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            pairs.append((iou_xyxy(p["bbox_xyxy"], g["bbox_xyxy"]), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for ov, pi, gi in pairs:
        if ov >= thr and pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
            out.append((pi, gi, ov))
    return out


def bbox_ap50_global(all_preds: List[Dict[str, Any]], gt_by_image: Dict[int, List[Dict[str, Any]]], thr: float = 0.5) -> float:
    total_gt = sum(len(v) for v in gt_by_image.values())
    if total_gt <= 0:
        return 0.0
    preds = sorted(all_preds, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    used: Dict[int, set[int]] = {img_id: set() for img_id in gt_by_image}
    tp: List[float] = []
    fp: List[float] = []
    for pred in preds:
        img_id = int(pred["image_id"])
        gts = gt_by_image.get(img_id, [])
        best_iou, best_idx = 0.0, -1
        for gi, gt in enumerate(gts):
            if gi in used.setdefault(img_id, set()):
                continue
            ov = iou_xyxy(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if ov > best_iou:
                best_iou, best_idx = ov, gi
        if best_iou >= thr and best_idx >= 0:
            used[img_id].add(best_idx)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    tp_cum = np.cumsum(np.asarray(tp, dtype=np.float64))
    fp_cum = np.cumsum(np.asarray(fp, dtype=np.float64))
    recalls = tp_cum / float(total_gt)
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    # VOC/COCO-style area under monotonic precision-recall envelope.
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return ap


def gt_from_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    img = sample["img"]
    img_id = int(img.get("id", img.get("img_id")))
    out = []
    for ann in sample["anns"]:
        x, y, w, h = map(float, ann["bbox"])
        valid = ann.get("valid") or {}
        gender_valid = bool(valid.get("gender", ann.get("gender") is not None))
        age_valid = bool(valid.get("age", ann.get("age") is not None))
        kpt_valid = bool(valid.get("keypoints_2d", False))
        keypoints = list(ann.get("keypoints") or [])
        if len(keypoints) < 51:
            keypoints += [0.0] * (51 - len(keypoints))
        out.append({
            "ann_id": int(ann.get("id", len(out))),
            "image_id": img_id,
            "bbox_xyxy": [x, y, x + w, y + h],
            "gender": ann.get("gender") if gender_valid else None,
            "age": ann.get("age") if age_valid else None,
            "keypoints": keypoints[:51] if kpt_valid else None,
        })
    return {"image_id": img_id, "boxes": out}


def normalize_gender(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"male", "m", "man", "1"}:
            return 1
        if low in {"female", "f", "woman", "0"}:
            return 0
        return None
    try:
        iv = int(round(float(value)))
    except Exception:
        return None
    return iv if iv in (0, 1) else None


def normalize_age(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def pose_pck(pred_kpts_xy: Sequence[Sequence[float]], gt_kpts_flat: Sequence[float], gt_box: Sequence[float], thr: float = 0.2) -> Tuple[int, int]:
    scale = max(1.0, math.sqrt(max(0.0, (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1]))))
    ok = total = 0
    for j in range(17):
        gi = 3 * j
        if gi + 2 >= len(gt_kpts_flat) or float(gt_kpts_flat[gi + 2]) <= 0:
            continue
        if j >= len(pred_kpts_xy):
            continue
        total += 1
        dx = float(pred_kpts_xy[j][0]) - float(gt_kpts_flat[gi])
        dy = float(pred_kpts_xy[j][1]) - float(gt_kpts_flat[gi + 1])
        if math.sqrt(dx * dx + dy * dy) <= thr * scale:
            ok += 1
    return ok, total


def _to_numpy(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_det_array(result: Dict[str, Any]) -> np.ndarray:
    # Supports both bbox_head.simple_test_bboxes output (Tensor/ndarray) and
    # detector.simple_test output where det is nested as [[class_array]].
    det = result.get("det")
    while isinstance(det, (list, tuple)) and len(det) == 1:
        det = det[0]
    if det is None:
        return np.zeros((0, 5), dtype=np.float32)
    arr = _to_numpy(det)
    if arr.ndim == 1 and arr.size == 0:
        return np.zeros((0, 5), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 5:
        return np.zeros((0, 5), dtype=np.float32)
    return arr.astype(np.float32, copy=False)


def _keypoints_xy_to_coco(kpts_xy: Sequence[Sequence[float]]) -> List[float]:
    out: List[float] = []
    for j in range(17):
        if j < len(kpts_xy):
            out.extend([float(kpts_xy[j][0]), float(kpts_xy[j][1]), 1.0])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def parse_stage2_result(result: Dict[str, Any], image_id: int, score_thr: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    arr_np = _extract_det_array(result)
    all_preds: List[Dict[str, Any]] = []
    thr_preds: List[Dict[str, Any]] = []
    for qi, row in enumerate(arr_np):
        score = float(row[4])
        kpts: List[List[float]] = []
        if row.shape[0] >= 4 + 1 + 1 + 1 + 34:
            tail = row[-34:]
            kpts = [[float(tail[2 * j]), float(tail[2 * j + 1])] for j in range(17)]
        pred = {
            "image_id": int(image_id),
            "query_index": int(qi),
            "bbox_xyxy": [float(row[0]), float(row[1]), float(row[2]), float(row[3])],
            "score": score,
            "gender": int(float(row[5]) >= 0.5) if row.shape[0] > 5 else None,
            "age": float(row[6]) if row.shape[0] > 6 else None,
            "keypoints": kpts,
        }
        all_preds.append(pred)
        if score >= score_thr:
            thr_preds.append(pred)
    return all_preds, thr_preds


def _extract_segmentation_records(result: Dict[str, Any], image_id: int, score_thr: float) -> List[Dict[str, Any]]:
    # Two supported forms:
    # 1) detector wrapper: result['ins_results'] = [(bbox_by_class, mask_by_class)]
    # 2) this mmdet3 port head: result['seg'] = [{'ins_results': (labels, bboxes, masks)}]
    records: List[Dict[str, Any]] = []

    seg = result.get("seg")
    if isinstance(seg, (list, tuple)) and seg:
        item = seg[0]
        if isinstance(item, dict) and "ins_results" in item:
            labels, bboxes, masks = item["ins_results"]
            labels_np = _to_numpy(labels).astype(np.int64, copy=False).reshape(-1)
            boxes_np = _to_numpy(bboxes)
            masks_np = _to_numpy(masks).astype(np.uint8, copy=False)
            if boxes_np.ndim == 2 and boxes_np.shape[1] >= 5:
                for idx, row in enumerate(boxes_np):
                    if idx >= len(labels_np) or idx >= len(masks_np):
                        continue
                    score = float(row[4])
                    if score < score_thr:
                        continue
                    records.append({
                        "image_id": int(image_id),
                        "category_id": int(labels_np[idx] + 1),
                        "score": score,
                        "segmentation": encode_binary_mask(masks_np[idx]),
                    })
                return records

    seg = result.get("ins_results")
    while isinstance(seg, (list, tuple)) and len(seg) == 1:
        seg = seg[0]
    if not isinstance(seg, (list, tuple)) or len(seg) != 2:
        return []
    det_by_class, masks_by_class = seg
    for class_idx, bboxes in enumerate(det_by_class):
        arr = _to_numpy(bboxes)
        if arr.ndim != 2 or arr.shape[1] < 5:
            continue
        masks = masks_by_class[class_idx] if class_idx < len(masks_by_class) else []
        for idx, row in enumerate(arr):
            score = float(row[4])
            if score < score_thr or idx >= len(masks):
                continue
            records.append({
                "image_id": int(image_id),
                "category_id": int(class_idx + 1),
                "score": score,
                "segmentation": encode_binary_mask(np.asarray(masks[idx], dtype=np.uint8)),
            })
    return records


def stage2_result_to_official_predictions(result: Dict[str, Any], image_id: int, score_thr: float) -> List[Dict[str, Any]]:
    _, thr_preds = parse_stage2_result(result, image_id, score_thr)
    records: List[Dict[str, Any]] = []
    for pred in thr_preds:
        x1, y1, x2, y2 = pred["bbox_xyxy"]
        rec: Dict[str, Any] = {
            "image_id": int(image_id),
            "category_id": 1,
            "bbox": [float(x1), float(y1), float(max(0.0, x2 - x1)), float(max(0.0, y2 - y1))],
            "score": float(pred["score"]),
        }
        if pred.get("keypoints"):
            rec["keypoints"] = _keypoints_xy_to_coco(pred["keypoints"])
        if pred.get("gender") is not None:
            rec["gender"] = int(pred["gender"])
        if pred.get("age") is not None:
            rec["age"] = float(pred["age"])
        records.append(rec)

    seg_records = _extract_segmentation_records(result, image_id, score_thr)
    if len(seg_records) == len(records):
        for rec, seg_rec in zip(records, seg_records):
            rec["segmentation"] = seg_rec["segmentation"]
    else:
        records.extend(seg_records)
    return records


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    cfg = make_cfg(argparse.Namespace(
        config=args.config,
        backbone_mode=args.backbone_mode,
        qwen_model=args.qwen_model,
        qwen_adapter=args.qwen_adapter,
        dino_model=args.dino_model,
        profile=args.profile,
        image_size=args.image_size,
        num_query=args.num_query,
        fusion_mode=args.fusion_mode,
        freeze_dino=args.freeze_dino,
    ))
    model = MODELS.build(cfg.model).to(device)
    model.eval()
    return model


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    if args.batch_size != 1:
        raise RuntimeError("This native port should be evaluated with --batch-size 1; simple_test only returns the first batch item.")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = load_index(args.val_ann_file, args.val_image_root, args.limit_images)
    model = load_model(args, device)
    # Initialize lazy layers exactly as training did, then load trainable checkpoint state.
    warm = collate_batches([build_batch(samples[0], args.image_size, args.max_instances, device)])
    with torch.no_grad():
        _ = model.extract_feat(warm["img"])
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_trainable", ckpt)
    incompatible = model.load_state_dict(state, strict=False)
    load_info = {
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "checkpoint_step": ckpt.get("step") if isinstance(ckpt, dict) else None,
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
    }
    print(json.dumps({"event": "checkpoint_loaded", **load_info}, ensure_ascii=False), flush=True)
    model.eval()

    pred_path = out_dir / "predictions.jsonl"
    official_pred_path = out_dir / "official_predictions.json"
    metrics_path = out_dir / "metrics.json"
    official_metrics_path = out_dir / "official_metrics.json"
    manifest_path = out_dir / "manifest.json"
    manifest = vars(args).copy()
    manifest.update({"num_samples": len(samples), "device": str(device), **load_info, "protocol": "stage2_native_224_single_scale_scorethr_diagnostic"})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    agg: Dict[str, Any] = {
        "images": 0,
        "gt_persons": 0,
        "pred_persons": 0,
        "pred_persons_all_queries": 0,
        "matched": 0,
        "pose_ok": 0,
        "pose_total": 0,
        "gender_gt_matched": 0,
        "gender_answered": 0,
        "gender_correct_answered": 0,
        "age_gt_matched": 0,
        "age_answered": 0,
        "age_abs_error_sum_answered": 0.0,
        "errors": 0,
    }
    gt_by_image: Dict[int, List[Dict[str, Any]]] = {}
    all_preds_for_ap: List[Dict[str, Any]] = []
    official_predictions: List[Dict[str, Any]] = []
    start = time.time()
    with pred_path.open("w", encoding="utf-8") as fout:
        for idx, sample in enumerate(samples):
            gt = gt_from_sample(sample)
            image_id = gt["image_id"]
            gt_by_image[image_id] = gt["boxes"]
            rec: Dict[str, Any] = {"image_id": image_id, "image": sample["img"].get("file_name"), "gt_persons": len(gt["boxes"])}
            try:
                batch = collate_batches([build_batch(sample, args.image_size, args.max_instances, device)])
                with torch.no_grad():
                    feat = model.extract_feat(batch["img"])
                    result = model.bbox_head.simple_test_bboxes(feat, batch["img_metas"], rescale=True)
                all_preds, thr_preds = parse_stage2_result(result, image_id, args.score_thr)
                all_preds_for_ap.extend(all_preds)
                official_predictions.extend(stage2_result_to_official_predictions(result, image_id, args.official_score_thr))
                matches = match_boxes(thr_preds, gt["boxes"], args.iou_thr)
                for pi, gi, ov in matches:
                    pred = thr_preds[pi]
                    g = gt["boxes"][gi]
                    if g.get("keypoints") is not None and pred.get("keypoints"):
                        ok, total = pose_pck(pred["keypoints"], g["keypoints"], g["bbox_xyxy"])
                        agg["pose_ok"] += ok
                        agg["pose_total"] += total
                    gg = normalize_gender(g.get("gender"))
                    pg = normalize_gender(pred.get("gender"))
                    if gg is not None:
                        agg["gender_gt_matched"] += 1
                        if pg is not None:
                            agg["gender_answered"] += 1
                            agg["gender_correct_answered"] += int(gg == pg)
                    ga = normalize_age(g.get("age"))
                    pa = normalize_age(pred.get("age"))
                    if ga is not None:
                        agg["age_gt_matched"] += 1
                        if pa is not None:
                            agg["age_answered"] += 1
                            agg["age_abs_error_sum_answered"] += abs(ga - pa)
                agg["images"] += 1
                agg["gt_persons"] += len(gt["boxes"])
                agg["pred_persons"] += len(thr_preds)
                agg["pred_persons_all_queries"] += len(all_preds)
                agg["matched"] += len(matches)
                rec.update({
                    "ok": True,
                    "pred_persons": len(thr_preds),
                    "pred_persons_all_queries": len(all_preds),
                    "matches": [[int(pi), int(gi), float(ov)] for pi, gi, ov in matches],
                    "predictions": thr_preds,
                })
            except Exception as exc:
                agg["images"] += 1
                agg["gt_persons"] += len(gt["boxes"])
                agg["errors"] += 1
                rec.update({"ok": False, "error": repr(exc), "predictions": []})
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if (idx + 1) % max(1, args.log_interval) == 0 or idx + 1 == len(samples):
                print(json.dumps({
                    "processed": idx + 1,
                    "matched": agg["matched"],
                    "pred_persons": agg["pred_persons"],
                    "errors": agg["errors"],
                    "elapsed_sec": round(time.time() - start, 3),
                }, ensure_ascii=False), flush=True)

    official_pred_path.write_text(json.dumps(official_predictions, ensure_ascii=False) + "\n", encoding="utf-8")
    official_metrics = evaluate_all_official(
        args.val_ann_file,
        official_predictions,
        image_ids=sorted(gt_by_image.keys()),
        attr_eval_source=Path(args.attr_eval_source),
    )
    official_metrics_path.write_text(json.dumps(official_metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = dict(agg)
    metrics.update({
        "score_thr": args.score_thr,
        "iou_thr": args.iou_thr,
        "bbox_AP50_global": bbox_ap50_global(all_preds_for_ap, gt_by_image, args.iou_thr),
        "box_recall_iou50": agg["matched"] / max(1, agg["gt_persons"]),
        "box_precision_iou50": agg["matched"] / max(1, agg["pred_persons"]),
        "pose_pck@0.2": agg["pose_ok"] / max(1, agg["pose_total"]),
        "gender_coverage_on_matched_valid_gt": agg["gender_answered"] / max(1, agg["gender_gt_matched"]),
        "gender_acc_answered_only": agg["gender_correct_answered"] / max(1, agg["gender_answered"]),
        "gender_acc_unknown_as_wrong": agg["gender_correct_answered"] / max(1, agg["gender_gt_matched"]),
        "age_coverage_on_matched_valid_gt": agg["age_answered"] / max(1, agg["age_gt_matched"]),
        "age_mae_answered_only": agg["age_abs_error_sum_answered"] / max(1, agg["age_answered"]),
        "elapsed_sec": round(time.time() - start, 3),
        "official_score_thr": args.official_score_thr,
        "official_prediction_count": len(official_predictions),
        "official_eval": official_metrics,
        "official_predictions": str(official_pred_path),
        "official_metrics_path": str(official_metrics_path),
        "protocol": "stage2_native_single_scale_with_official_cocoeval_and_humanquerynet_attr",
        "note": "This evaluates the Stage2 native single-scale path and reports official COCOeval/AttrCOCOeval metrics from exported predictions. It is still not an 800/1333 multi-scale MMDet test pipeline.",
    })
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
