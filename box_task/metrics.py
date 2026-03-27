from typing import Any, Dict, List, Optional, Tuple


def _to_xyxy(box_obj: Any) -> Optional[List[float]]:
    if isinstance(box_obj, dict):
        box = box_obj.get("bbox_xyxy", box_obj.get("bbox", None))
    else:
        box = box_obj
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        return [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    except Exception:
        return None


def _iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 1e-8:
        return 0.0
    return inter / union


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _match_boxes(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    iou_thr: float = 0.5,
) -> Tuple[int, int, int, float]:
    used = [False] * len(pred_boxes)
    tp = 0
    iou_sum = 0.0
    for g in gt_boxes:
        best_j = -1
        best_iou = 0.0
        for j, p in enumerate(pred_boxes):
            if used[j]:
                continue
            iou = _iou(g, p)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0 and best_iou >= iou_thr:
            used[best_j] = True
            tp += 1
            iou_sum += best_iou
    fp = max(0, len(pred_boxes) - tp)
    fn = max(0, len(gt_boxes) - tp)
    mean_iou = _safe_div(iou_sum, tp)
    return tp, fp, fn, mean_iou


def evaluate_box_records(records: List[Dict[str, Any]], iou_thr: float = 0.5) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {}

    valid_json = 0
    tp = fp = fn = 0
    matched_iou_sum = 0.0
    matched_img_cnt = 0
    box_count_err = 0.0

    for rec in records:
        gt = rec.get("gt_struct", {}) or {}
        pred = rec.get("pred_struct", {}) or {}
        if rec.get("valid_json", False):
            valid_json += 1

        gt_boxes = [_to_xyxy(x) for x in gt.get("box", [])]
        gt_boxes = [b for b in gt_boxes if b is not None]
        pred_boxes = [_to_xyxy(x) for x in pred.get("box", [])]
        pred_boxes = [b for b in pred_boxes if b is not None]

        box_count_err += abs(len(gt_boxes) - len(pred_boxes))
        _tp, _fp, _fn, _miou = _match_boxes(gt_boxes, pred_boxes, iou_thr=iou_thr)
        tp += _tp
        fp += _fp
        fn += _fn
        if _tp > 0:
            matched_iou_sum += _miou
            matched_img_cnt += 1

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    return {
        "valid_json_rate": valid_json / total,
        "box_count_mae": box_count_err / total,
        "box_precision_iou50": precision,
        "box_recall_iou50": recall,
        "box_f1_iou50": f1,
        "box_mean_iou_on_matched_images": _safe_div(matched_iou_sum, matched_img_cnt),
    }
