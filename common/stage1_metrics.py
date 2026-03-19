from typing import Any, Dict, List, Optional, Tuple


def _box_to_xyxy(box_obj: Any) -> Optional[List[float]]:
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


def _match_boxes(gt_boxes: List[List[float]], pred_boxes: List[List[float]], iou_thr: float = 0.5) -> Tuple[int, int, int, float]:
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
    mean_iou = iou_sum / tp if tp > 0 else 0.0
    return tp, fp, fn, mean_iou


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def _extract_pose_list(obj: Dict[str, Any]) -> List[Any]:
    poses = obj.get("pose", [])
    if isinstance(poses, list):
        return poses
    return []


def _extract_parsing_labels(obj: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for item in obj.get("parsing", []):
        if not isinstance(item, dict):
            continue
        part_labels = item.get("labels", [])
        if isinstance(part_labels, list):
            for x in part_labels:
                labels.append(str(x))
    return labels


def _pose_distance(gt_pose: Any, pred_pose: Any) -> Optional[float]:
    if not isinstance(gt_pose, dict) or not isinstance(pred_pose, dict):
        return None
    gt_kpts = gt_pose.get("keypoints", [])
    pred_kpts = pred_pose.get("keypoints", [])
    if not isinstance(gt_kpts, list) or not isinstance(pred_kpts, list):
        return None
    n = min(len(gt_kpts), len(pred_kpts))
    if n == 0:
        return None
    dist_sum = 0.0
    cnt = 0
    for i in range(n):
        g = gt_kpts[i]
        p = pred_kpts[i]
        if not isinstance(g, (list, tuple)) or not isinstance(p, (list, tuple)):
            continue
        if len(g) < 2 or len(p) < 2:
            continue
        gx, gy = float(g[0]), float(g[1])
        px, py = float(p[0]), float(p[1])
        dx = gx - px
        dy = gy - py
        dist_sum += (dx * dx + dy * dy) ** 0.5
        cnt += 1
    if cnt == 0:
        return None
    return dist_sum / cnt


def evaluate_stage1_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    total = len(records)
    if total == 0:
        return {}

    valid_json = 0
    sum_abs_box_count_err = 0.0
    tp = fp = fn = 0
    sum_iou = 0.0
    matched_images = 0
    pose_dist_sum = 0.0
    pose_dist_cnt = 0
    parsing_tp = parsing_fp = parsing_fn = 0

    for rec in records:
        gt = rec.get("gt_struct", {}) or {}
        pred = rec.get("pred_struct", {}) or {}
        if rec.get("valid_json", False):
            valid_json += 1

        gt_boxes = [_box_to_xyxy(x) for x in gt.get("box", [])]
        gt_boxes = [b for b in gt_boxes if b is not None]
        pred_boxes = [_box_to_xyxy(x) for x in pred.get("box", [])]
        pred_boxes = [b for b in pred_boxes if b is not None]

        sum_abs_box_count_err += abs(len(gt_boxes) - len(pred_boxes))

        _tp, _fp, _fn, miou = _match_boxes(gt_boxes, pred_boxes, iou_thr=0.5)
        tp += _tp
        fp += _fp
        fn += _fn
        if _tp > 0:
            sum_iou += miou
            matched_images += 1

        gt_pose_list = _extract_pose_list(gt)
        pred_pose_list = _extract_pose_list(pred)
        for i in range(min(len(gt_pose_list), len(pred_pose_list))):
            d = _pose_distance(gt_pose_list[i], pred_pose_list[i])
            if d is not None:
                pose_dist_sum += d
                pose_dist_cnt += 1

        gt_labels = set(_extract_parsing_labels(gt))
        pred_labels = set(_extract_parsing_labels(pred))
        parsing_tp += len(gt_labels & pred_labels)
        parsing_fp += len(pred_labels - gt_labels)
        parsing_fn += len(gt_labels - pred_labels)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    box_f1 = _safe_div(2 * precision * recall, precision + recall)

    parsing_precision = _safe_div(parsing_tp, parsing_tp + parsing_fp)
    parsing_recall = _safe_div(parsing_tp, parsing_tp + parsing_fn)
    parsing_f1 = _safe_div(
        2 * parsing_precision * parsing_recall, parsing_precision + parsing_recall
    )

    return {
        "valid_json_rate": valid_json / total,
        "box_count_mae": sum_abs_box_count_err / total,
        "box_precision_iou50": precision,
        "box_recall_iou50": recall,
        "box_f1_iou50": box_f1,
        "box_mean_iou_on_matched_images": _safe_div(sum_iou, matched_images),
        "pose_mean_l2_pixel": _safe_div(pose_dist_sum, pose_dist_cnt),
        "parsing_label_f1": parsing_f1,
    }
