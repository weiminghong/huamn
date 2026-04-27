import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from qwen4b_coco_unihuman_ft.eval_scheme2 import (
    box_iou,
    evaluate_records_scheme2,
    extract_prediction_struct,
    get_record_gt,
    load_jsonl,
)


def _image_size_from_record(record: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    if isinstance(record.get("image_size"), list) and len(record["image_size"]) >= 2:
        return int(record["image_size"][0]), int(record["image_size"][1])
    if isinstance(record.get("width"), int) and isinstance(record.get("height"), int):
        return int(record["width"]), int(record["height"])
    return None


def _best_iou_summary(gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for pi, pred_box in enumerate(pred_boxes):
        best_iou = 0.0
        best_gt = -1
        for gi, gt_box in enumerate(gt_boxes):
            iou = box_iou(gt_box["bbox_xyxy"], pred_box["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt = gi
        rows.append({"pred_index": pi, "best_gt_index": best_gt, "best_iou": best_iou})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect parsed predictions and scheme-2 matching.")
    parser.add_argument("--gt-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--pck-thr", type=float, default=0.2)
    parser.add_argument("--print-limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_records = load_jsonl(args.pred_jsonl)
    pred_ids = {str(r.get("id", r.get("image_id", ""))) for r in pred_records}
    gt_records = [
        r
        for r in load_jsonl(args.gt_jsonl)
        if str(r.get("id", r.get("image_id", ""))) in pred_ids
    ]
    pred_by_id = {str(r.get("id", r.get("image_id", ""))): r for r in pred_records}

    rows: List[Dict[str, Any]] = []
    for gt_rec in gt_records:
        sid = str(gt_rec.get("id", gt_rec.get("image_id", "")))
        pred_rec = pred_by_id.get(sid, {"id": sid})
        image_size = _image_size_from_record(pred_rec) or _image_size_from_record(gt_rec)
        gt = get_record_gt(gt_rec, image_size=image_size)
        pred, valid = extract_prediction_struct(pred_rec, image_size=image_size)
        rows.append(
            {
                "id": sid,
                "image": gt_rec.get("image", pred_rec.get("image", "")),
                "valid_prediction": valid,
                "gt_box": len(gt["box"]),
                "pred_box": len(pred["box"]),
                "pred_best_ious": _best_iou_summary(gt["box"], pred["box"]),
                "prediction_text_head": str(pred_rec.get("prediction_text", ""))[:500],
            }
        )

    metrics = evaluate_records_scheme2(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_thr=float(args.iou_thr),
        pck_thr=float(args.pck_thr),
    )
    payload = {"metrics": metrics, "images": rows}
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(json.dumps({k: v for k, v in metrics.items() if k != "per_image"}, ensure_ascii=False, indent=2))
    for row in rows[: int(args.print_limit)]:
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
