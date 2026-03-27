from typing import Any, Dict, List, Optional


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def extract_pred_boxes(
    pred_struct: Dict[str, Any],
    score_thr: float = 0.0,
    image_size: Optional[List[int]] = None,
) -> List[Dict[str, float]]:
    boxes = pred_struct.get("box", [])
    if not isinstance(boxes, list):
        return []

    width = height = None
    if image_size is not None and len(image_size) == 2:
        width, height = image_size

    parsed = []
    for item in boxes:
        if not isinstance(item, dict):
            continue
        raw_box = item.get("bbox_xyxy", item.get("bbox"))
        if not isinstance(raw_box, (list, tuple)) or len(raw_box) != 4:
            continue
        x1, y1, x2, y2 = [_to_float(v) for v in raw_box]
        if width is not None:
            x1 = max(0.0, min(x1, width - 1))
            x2 = max(0.0, min(x2, width - 1))
        if height is not None:
            y1 = max(0.0, min(y1, height - 1))
            y2 = max(0.0, min(y2, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        score = _to_float(item.get("score", 1.0), 1.0)
        if score < score_thr:
            continue
        parsed.append({"bbox_xyxy": [x1, y1, x2, y2], "score": score})
    return parsed


def iou(box_a: List[float], box_b: List[float]) -> float:
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
    return 0.0 if union <= 1e-8 else inter / union


def nms(
    boxes: List[Dict[str, Any]],
    iou_thr: float = 0.5,
    max_per_img: int = 100,
) -> List[Dict[str, Any]]:
    if not boxes:
        return []
    order = sorted(range(len(boxes)), key=lambda i: boxes[i].get("score", 1.0), reverse=True)
    keep = []
    while order and len(keep) < max_per_img:
        i = order.pop(0)
        current = boxes[i]
        keep.append(current)
        current_box = current["bbox_xyxy"]
        remain = []
        for j in order:
            if iou(current_box, boxes[j]["bbox_xyxy"]) <= iou_thr:
                remain.append(j)
        order = remain
    return keep
