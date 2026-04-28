import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


COCO_SIGMAS = [
    0.26,
    0.25,
    0.25,
    0.35,
    0.35,
    0.79,
    0.79,
    0.72,
    0.72,
    0.62,
    0.62,
    1.07,
    1.07,
    0.87,
    0.87,
    0.89,
    0.89,
]
COCO_SIGMAS = [x / 10.0 for x in COCO_SIGMAS]


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def dump_json(path: str, obj: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    for chunk in fenced:
        try:
            obj = json.loads(chunk)
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = text[start : i + 1]
                try:
                    obj = json.loads(chunk)
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
    return None


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _clip_box(box: List[float], image_size: Optional[Tuple[int, int]]) -> List[float]:
    x1, y1, x2, y2 = box
    if image_size is not None:
        w, h = image_size
        x1 = max(0.0, min(x1, float(max(0, w - 1))))
        x2 = max(0.0, min(x2, float(max(0, w - 1))))
        y1 = max(0.0, min(y1, float(max(0, h - 1))))
        y2 = max(0.0, min(y2, float(max(0, h - 1))))
    return [x1, y1, x2, y2]


def _is_norm_1000_format(fmt: str) -> bool:
    fmt = str(fmt).lower()
    return "norm" in fmt or "1000" in fmt


def normalize_box_item(item: Any, image_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw = item.get("bbox_2d", item.get("bbox_xyxy", item.get("xyxy", item.get("bbox", item.get("box")))))
        score = _to_float(item.get("score", 1.0))
        person_id = item.get("person_id", None)
        fmt = str(item.get("bbox_format", item.get("format", ""))).lower()
    else:
        raw = item
        score = 1.0
        person_id = None
        fmt = ""
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    vals = [_to_float(v) for v in raw[:4]]
    if any(v is None for v in vals):
        return None
    x1, y1, x2, y2 = [float(v) for v in vals]
    if "xywh" in fmt:
        x2 = x1 + max(0.0, x2)
        y2 = y1 + max(0.0, y2)
    if image_size is not None and _is_norm_1000_format(fmt):
        w, h = image_size
        x1 = x1 / 1000.0 * float(w)
        x2 = x2 / 1000.0 * float(w)
        y1 = y1 / 1000.0 * float(h)
        y2 = y2 / 1000.0 * float(h)
    if x2 <= x1 or y2 <= y1:
        return None
    out = {
        "person_id": person_id,
        "bbox_xyxy": _clip_box([x1, y1, x2, y2], image_size),
        "score": 1.0 if score is None else max(0.0, min(float(score), 1.0)),
    }
    return out


def normalize_boxes(items: Any, image_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    boxes: List[Dict[str, Any]] = []
    for item in items:
        box = normalize_box_item(item, image_size=image_size)
        if box is not None:
            boxes.append(box)
    return boxes


def normalize_pose_item(item: Any, image_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    if isinstance(item, dict):
        raw = item.get("keypoints_2d", item.get("keypoints", item.get("kpts", [])))
        score = _to_float(item.get("score", 1.0))
        person_id = item.get("person_id", None)
        fmt = str(item.get("keypoint_format", item.get("kpt_format", item.get("format", "")))).lower()
    else:
        raw = item
        score = 1.0
        person_id = None
        fmt = ""

    triplets: List[List[float]] = []
    if isinstance(raw, list) and raw and all(not isinstance(x, (list, tuple)) for x in raw):
        n = min(17, len(raw) // 3)
        for i in range(n):
            x = _to_float(raw[3 * i + 0])
            y = _to_float(raw[3 * i + 1])
            v = _to_float(raw[3 * i + 2])
            if x is None or y is None or v is None:
                return None
            triplets.append([float(x), float(y), float(v)])
    elif isinstance(raw, list):
        for kp in raw[:17]:
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                return None
            x = _to_float(kp[0])
            y = _to_float(kp[1])
            v = _to_float(kp[2])
            if x is None or y is None or v is None:
                return None
            triplets.append([float(x), float(y), float(v)])
    if len(triplets) != 17:
        return None

    if image_size is not None:
        w, h = image_size
        for kp in triplets:
            if kp[2] <= 0:
                continue
            if _is_norm_1000_format(fmt):
                kp[0] = kp[0] / 1000.0 * float(w)
                kp[1] = kp[1] / 1000.0 * float(h)
            kp[0] = max(0.0, min(kp[0], float(max(0, w - 1))))
            kp[1] = max(0.0, min(kp[1], float(max(0, h - 1))))
    return {
        "person_id": person_id,
        "keypoints": triplets,
        "score": 1.0 if score is None else max(0.0, min(float(score), 1.0)),
    }


def normalize_poses(items: Any, image_size: Optional[Tuple[int, int]] = None) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    poses: List[Dict[str, Any]] = []
    for item in items:
        pose = normalize_pose_item(item, image_size=image_size)
        if pose is not None:
            poses.append(pose)
    return poses


def normalize_attributes(items: Any) -> List[Dict[str, Any]]:
    if not isinstance(items, list):
        return []
    attrs: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out: Dict[str, Any] = {"person_id": item.get("person_id", None)}
        if "gender" in item:
            g = _to_float(item.get("gender"))
            if g is not None and int(round(g)) in (0, 1):
                out["gender"] = int(round(g))
        if "age" in item:
            a = _to_float(item.get("age"))
            if a is not None and a >= 0:
                out["age"] = float(a)
        if len(out) > 1:
            attrs.append(out)
    return attrs


def _zero_pose_17() -> List[List[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(17)]


def _align_box_pose(struct_obj: Dict[str, Any]) -> Dict[str, Any]:
    boxes = struct_obj.get("box", [])
    poses = struct_obj.get("pose", [])
    if not isinstance(boxes, list):
        boxes = []
    if not isinstance(poses, list):
        poses = []

    aligned_poses: List[Dict[str, Any]] = []
    for i, box_item in enumerate(boxes):
        score = 1.0
        if isinstance(box_item, dict):
            score = _to_float(box_item.get("score", 1.0)) or 1.0
        if i < len(poses) and isinstance(poses[i], dict):
            kpts = poses[i].get("keypoints", [])
            if isinstance(kpts, list) and len(kpts) == 17:
                aligned_poses.append({"person_id": box_item.get("person_id") if isinstance(box_item, dict) else None, "keypoints": kpts, "score": score})
                continue
        aligned_poses.append({"person_id": box_item.get("person_id") if isinstance(box_item, dict) else None, "keypoints": _zero_pose_17(), "score": score})
    return {"box": boxes, "pose": aligned_poses, "attribute": struct_obj.get("attribute", []) if isinstance(struct_obj.get("attribute", []), list) else []}


def _truncate_to_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    start_m = re.search(r"<answer>", text, flags=re.IGNORECASE)
    if start_m:
        text = text[start_m.start():]
    end_m = re.search(r"</answer>", text, flags=re.IGNORECASE)
    if end_m:
        text = text[: end_m.end()]
    return text


def parse_tag_struct(text: str, image_size: Optional[Tuple[int, int]] = None) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if "No person detected" in text:
        return {"box": [], "pose": [], "attribute": []}
    if image_size is None:
        return None

    text = _truncate_to_answer(text)
    width, height = image_size
    person_blobs = re.split(r"<person>", text, flags=re.IGNORECASE)[1:]
    if not person_blobs:
        if re.search(r"<answer>\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE):
            return {"box": [], "pose": [], "attribute": []}
        return None

    boxes: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    attrs: List[Dict[str, Any]] = []
    num_pattern = r"-?(?:\d+\.\d+|\d+|\.\d+)"
    for person_id, blob in enumerate(person_blobs, start=1):
        score = 1.0
        m_score = re.search(r"<score>\s*([0-9]*\.?[0-9]+)", blob, flags=re.IGNORECASE)
        if m_score:
            score = _to_float(m_score.group(1)) or 1.0
        score = max(0.0, min(score, 1.0))

        m_box = re.search(r"<box>(.*?)(?:</box>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_box is None:
            continue
        box_nums = re.findall(num_pattern, m_box.group(1))
        if len(box_nums) < 4:
            continue
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = [float(n) for n in box_nums[:4]]
        except Exception:
            continue
        x1 = max(0.0, min((x1_norm / 1000.0) * width, width - 1))
        y1 = max(0.0, min((y1_norm / 1000.0) * height, height - 1))
        x2 = max(0.0, min((x2_norm / 1000.0) * width, width - 1))
        y2 = max(0.0, min((y2_norm / 1000.0) * height, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        boxes.append({"person_id": person_id, "bbox_xyxy": [x1, y1, x2, y2], "score": score})

        m_pose = re.search(r"<pose>(.*?)(?:</pose>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        kpts: List[List[float]] = []
        if m_pose is not None:
            pose_nums = re.findall(num_pattern, m_pose.group(1))
            if len(pose_nums) >= 51:
                for i in range(17):
                    try:
                        x_n = float(pose_nums[i * 3])
                        y_n = float(pose_nums[i * 3 + 1])
                        v_n = int(round(float(pose_nums[i * 3 + 2])))
                    except Exception:
                        x_n, y_n, v_n = 0.0, 0.0, 0
                    if x_n == 0 and y_n == 0 and v_n == 0:
                        kpts.append([0.0, 0.0, 0.0])
                    else:
                        x = max(0.0, min((x_n / 1000.0) * width, width - 1))
                        y = max(0.0, min((y_n / 1000.0) * height, height - 1))
                        if v_n not in (0, 1, 2):
                            v_n = 0
                        kpts.append([x, y, float(v_n)])
            elif len(pose_nums) >= 34:
                for i in range(17):
                    try:
                        x_n = float(pose_nums[i * 2])
                        y_n = float(pose_nums[i * 2 + 1])
                    except Exception:
                        x_n, y_n = 0.0, 0.0
                    if x_n == 0 and y_n == 0:
                        kpts.append([0.0, 0.0, 0.0])
                    else:
                        x = max(0.0, min((x_n / 1000.0) * width, width - 1))
                        y = max(0.0, min((y_n / 1000.0) * height, height - 1))
                        kpts.append([x, y, 2.0])

        poses.append({"person_id": person_id, "keypoints": kpts if len(kpts) == 17 else _zero_pose_17(), "score": score})
        attr_item: Dict[str, Any] = {"person_id": person_id}

        m_gender = re.search(r"<gender>(.*?)(?:</gender>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_gender is not None:
            gender_text = m_gender.group(1).strip().lower()
            gender_val: Optional[int] = None
            if gender_text in ("male", "man", "m"):
                gender_val = 0
            elif gender_text in ("female", "woman", "f"):
                gender_val = 1
            else:
                gender_num = _to_float(gender_text)
                if gender_num is not None:
                    gender_val = int(round(gender_num))
            if gender_val in (0, 1):
                attr_item["gender"] = gender_val

        m_age = re.search(r"<age>(.*?)(?:</age>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_age is not None:
            age_nums = re.findall(num_pattern, m_age.group(1))
            if age_nums:
                age_val = _to_float(age_nums[0])
                if age_val is not None and age_val >= 0:
                    attr_item["age"] = float(age_val)

        if len(attr_item) > 1:
            attrs.append(attr_item)

    return _align_box_pose({"box": boxes, "pose": poses, "attribute": attrs}) if boxes else None


def _is_flat_box(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) >= 4
        and all(not isinstance(x, (list, tuple, dict)) for x in value[:4])
    )


def _extract_box_items(obj: Dict[str, Any]) -> Any:
    for key in ("box", "boxes", "bbox", "bboxes", "detections"):
        value = obj.get(key, None)
        if _is_flat_box(value):
            return [value]
        if isinstance(value, list):
            return value
    persons = obj.get("persons", obj.get("person", None))
    if isinstance(persons, list):
        return persons
    return []


def _extract_pose_items(obj: Dict[str, Any]) -> Any:
    for key in ("pose", "poses", "keypoints", "kpts"):
        value = obj.get(key, None)
        if isinstance(value, list):
            return value
    persons = obj.get("persons", obj.get("person", None))
    if isinstance(persons, list):
        return persons
    return []


def _extract_attribute_items(obj: Dict[str, Any]) -> Any:
    for key in ("attribute", "attributes", "attrs"):
        value = obj.get(key, None)
        if isinstance(value, list):
            return value
    persons = obj.get("persons", obj.get("person", None))
    if isinstance(persons, list):
        return persons
    return []


def normalize_struct(obj: Optional[Dict[str, Any]], image_size: Optional[Tuple[int, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
    obj = obj if isinstance(obj, dict) else {}
    return {
        "box": normalize_boxes(_extract_box_items(obj), image_size=image_size),
        "pose": normalize_poses(_extract_pose_items(obj), image_size=image_size),
        "attribute": normalize_attributes(_extract_attribute_items(obj)),
    }


def extract_prediction_struct(record: Dict[str, Any], image_size: Optional[Tuple[int, int]] = None) -> Tuple[Dict[str, Any], bool]:
    if record.get("valid_struct", True) is False:
        return normalize_struct(None, image_size=image_size), False
    for key in ("prediction_struct_post", "prediction_struct_raw", "prediction", "pred"):
        obj = record.get(key, None)
        if isinstance(obj, dict):
            return normalize_struct(obj, image_size=image_size), True
    for key in ("prediction_text", "text", "response", "output"):
        text = str(record.get(key, ""))
        tag_obj = parse_tag_struct(text, image_size=image_size)
        if tag_obj is not None:
            return normalize_struct(tag_obj, image_size=image_size), True
        obj = parse_json_object_from_text(text)
        if obj is not None:
            return normalize_struct(obj, image_size=image_size), True
    return normalize_struct(None, image_size=image_size), False


def get_record_gt(record: Dict[str, Any], image_size: Optional[Tuple[int, int]] = None) -> Dict[str, List[Dict[str, Any]]]:
    if isinstance(record.get("gt_struct"), dict):
        return normalize_struct(record["gt_struct"], image_size=image_size)
    return normalize_struct(
        {
            "box": record.get("box", []),
            "pose": record.get("pose", []),
            "attribute": record.get("attribute", []),
        },
        image_size=image_size,
    )


def box_area(box: List[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def box_iou(a: List[float], b: List[float]) -> float:
    ix1 = max(float(a[0]), float(b[0]))
    iy1 = max(float(a[1]), float(b[1]))
    ix2 = min(float(a[2]), float(b[2]))
    iy2 = min(float(a[3]), float(b[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = box_area(a) + box_area(b) - inter
    return 0.0 if union <= 0 else inter / union


def greedy_match_boxes(
    gt_boxes: List[Dict[str, Any]],
    pred_boxes: List[Dict[str, Any]],
    iou_thr: float,
) -> List[Tuple[int, int, float]]:
    pairs: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou >= float(iou_thr):
                pairs.append((iou, gi, pi))
    pairs.sort(reverse=True, key=lambda x: x[0])
    used_g = set()
    used_p = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, gi, pi in pairs:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        matches.append((gi, pi, iou))
    return matches


def bbox_ap_at_iou(gt_boxes: List[Dict[str, Any]], pred_boxes: List[Dict[str, Any]], iou_thr: float) -> float:
    if not gt_boxes:
        return 0.0
    preds = sorted(enumerate(pred_boxes), key=lambda x: float(x[1].get("score", 1.0)), reverse=True)
    used_gt = set()
    tp: List[float] = []
    fp: List[float] = []
    for _, pred in preds:
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in used_gt:
                continue
            iou = box_iou(gt["bbox_xyxy"], pred["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        if best_iou >= float(iou_thr) and best_gi >= 0:
            used_gt.add(best_gi)
            tp.append(1.0)
            fp.append(0.0)
        else:
            tp.append(0.0)
            fp.append(1.0)
    if not tp:
        return 0.0
    cum_tp: List[float] = []
    cum_fp: List[float] = []
    s_tp = 0.0
    s_fp = 0.0
    for t, f in zip(tp, fp):
        s_tp += t
        s_fp += f
        cum_tp.append(s_tp)
        cum_fp.append(s_fp)
    recalls = [x / max(1, len(gt_boxes)) for x in cum_tp]
    precisions = [cum_tp[i] / max(1e-12, cum_tp[i] + cum_fp[i]) for i in range(len(cum_tp))]
    ap = 0.0
    for r in [x / 100.0 for x in range(0, 101)]:
        p = max([precisions[i] for i, rec in enumerate(recalls) if rec >= r] or [0.0])
        ap += p / 101.0
    return float(ap)


def _by_person_id(items: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {x.get("person_id"): x for x in items if x.get("person_id", None) is not None}


def get_aligned_item(items: List[Dict[str, Any]], boxes: List[Dict[str, Any]], box_index: int) -> Optional[Dict[str, Any]]:
    if box_index < 0 or box_index >= len(boxes):
        return None
    pid = boxes[box_index].get("person_id", None)
    if pid is not None:
        hit = _by_person_id(items).get(pid)
        if hit is not None:
            return hit
    if box_index < len(items):
        return items[box_index]
    return None


def has_valid_pose(pose: Dict[str, Any]) -> bool:
    kpts = pose.get("keypoints", [])
    return isinstance(kpts, list) and any(len(kp) >= 3 and float(kp[2]) > 0 for kp in kpts)


def visible_keypoint_count(pose: Dict[str, Any]) -> int:
    kpts = pose.get("keypoints", [])
    if not isinstance(kpts, list):
        return 0
    return sum(
        1
        for kp in kpts[:17]
        if isinstance(kp, (list, tuple)) and len(kp) >= 3 and float(kp[2]) > 0
    )


def pose_pck_and_oks(gt_pose: Dict[str, Any], pred_pose: Optional[Dict[str, Any]], gt_box: Dict[str, Any], pck_thr: float) -> Tuple[int, int, float]:
    gt_kpts = gt_pose.get("keypoints", [])
    pred_kpts = pred_pose.get("keypoints", []) if isinstance(pred_pose, dict) else []
    scale = math.sqrt(max(box_area(gt_box["bbox_xyxy"]), 1.0))
    correct = 0
    total = 0
    oks_sum = 0.0
    oks_n = 0
    for i, gt_kp in enumerate(gt_kpts[:17]):
        if not isinstance(gt_kp, (list, tuple)) or len(gt_kp) < 3 or float(gt_kp[2]) <= 0:
            continue
        total += 1
        if not isinstance(pred_kpts, list) or i >= len(pred_kpts):
            continue
        pred_kp = pred_kpts[i]
        if not isinstance(pred_kp, (list, tuple)) or len(pred_kp) < 3 or float(pred_kp[2]) <= 0:
            continue
        dx = float(pred_kp[0]) - float(gt_kp[0])
        dy = float(pred_kp[1]) - float(gt_kp[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist <= float(pck_thr) * scale:
            correct += 1
        sigma = COCO_SIGMAS[i]
        var = (2.0 * sigma) ** 2
        area = max(box_area(gt_box["bbox_xyxy"]), 1.0)
        oks_sum += math.exp(-(dx * dx + dy * dy) / (2.0 * area * var + 1e-12))
        oks_n += 1
    mean_oks = oks_sum / oks_n if oks_n > 0 else 0.0
    return correct, total, mean_oks


def evaluate_records_scheme2(
    gt_records: Iterable[Dict[str, Any]],
    pred_records: Iterable[Dict[str, Any]],
    iou_thr: float = 0.5,
    pck_thr: float = 0.2,
) -> Dict[str, Any]:
    gt_by_id = {str(r.get("id", r.get("image_id", ""))): r for r in gt_records}
    pred_by_id = {str(r.get("id", r.get("image_id", ""))): r for r in pred_records}
    ids = list(gt_by_id.keys())

    n_gt_box = 0
    n_pred_box = 0
    tp_box = 0
    sum_ap50 = 0.0
    sum_ap5095 = 0.0
    valid_json = 0
    pose_gt_valid_total = 0
    pose_visible_kpt_total_all_gt = 0
    pose_gt_valid = 0
    pose_kpt_correct = 0
    pose_kpt_total = 0
    pose_oks_values: List[float] = []
    gender_total = 0
    gender_correct = 0
    gender_missing_pred = 0
    age_total = 0
    age_abs_err_sum = 0.0
    age_acc10 = 0
    age_missing_pred = 0
    per_image: List[Dict[str, Any]] = []

    for sid in ids:
        gt_rec = gt_by_id[sid]
        pred_rec = pred_by_id.get(sid, {"id": sid})
        image_size = None
        if isinstance(gt_rec.get("width"), int) and isinstance(gt_rec.get("height"), int):
            image_size = (int(gt_rec["width"]), int(gt_rec["height"]))
        gt = get_record_gt(gt_rec, image_size=image_size)
        pred, ok = extract_prediction_struct(pred_rec, image_size=image_size)
        valid_json += int(ok)

        gt_boxes = gt["box"]
        pred_boxes = pred["box"]
        matches = greedy_match_boxes(gt_boxes, pred_boxes, iou_thr=float(iou_thr))
        matched_g = {gi for gi, _, _ in matches}
        matched_p = {pi for _, pi, _ in matches}
        best_ious = []
        for pi, pred_box in enumerate(pred_boxes):
            best_iou = 0.0
            best_gt = -1
            for gi, gt_box in enumerate(gt_boxes):
                iou = box_iou(gt_box["bbox_xyxy"], pred_box["bbox_xyxy"])
                if iou > best_iou:
                    best_iou = iou
                    best_gt = gi
            best_ious.append({"pred_index": pi, "best_gt_index": best_gt, "best_iou": float(best_iou)})

        n_gt_box += len(gt_boxes)
        n_pred_box += len(pred_boxes)
        tp_box += len(matches)
        sum_ap50 += bbox_ap_at_iou(gt_boxes, pred_boxes, 0.5)
        thrs = [0.5 + 0.05 * i for i in range(10)]
        sum_ap5095 += sum(bbox_ap_at_iou(gt_boxes, pred_boxes, t) for t in thrs) / len(thrs)

        gt_pose_by_pid = _by_person_id(gt["pose"])
        pred_poses = pred["pose"]
        gt_attr_by_pid = _by_person_id(gt["attribute"])
        pred_attrs = pred["attribute"]

        for gi, gt_box in enumerate(gt_boxes):
            gt_pid = gt_box.get("person_id", None)
            gt_pose = gt_pose_by_pid.get(gt_pid) if gt_pid is not None else None
            if gt_pose is None and gi < len(gt["pose"]):
                gt_pose = gt["pose"][gi]
            if gt_pose is not None and has_valid_pose(gt_pose):
                pose_gt_valid_total += 1
                pose_visible_kpt_total_all_gt += visible_keypoint_count(gt_pose)

        img_pose_gt_valid = 0
        img_pose_kpt_total = 0
        img_pose_kpt_correct = 0
        for gi, pi, _ in matches:
            gt_box = gt_boxes[gi]
            gt_pid = gt_box.get("person_id", None)
            gt_pose = gt_pose_by_pid.get(gt_pid) if gt_pid is not None else None
            if gt_pose is None and gi < len(gt["pose"]):
                gt_pose = gt["pose"][gi]
            if gt_pose is not None and has_valid_pose(gt_pose):
                pose_gt_valid += 1
                img_pose_gt_valid += 1
                pred_pose = get_aligned_item(pred_poses, pred_boxes, pi)
                c, t, oks = pose_pck_and_oks(gt_pose, pred_pose, gt_box, pck_thr=float(pck_thr))
                pose_kpt_correct += c
                pose_kpt_total += t
                img_pose_kpt_correct += c
                img_pose_kpt_total += t
                if t > 0:
                    pose_oks_values.append(oks)

            gt_attr = gt_attr_by_pid.get(gt_pid) if gt_pid is not None else None
            if gt_attr is None and gi < len(gt["attribute"]):
                gt_attr = gt["attribute"][gi]
            pred_attr = get_aligned_item(pred_attrs, pred_boxes, pi)
            if isinstance(gt_attr, dict) and "gender" in gt_attr:
                gender_total += 1
                if not isinstance(pred_attr, dict) or "gender" not in pred_attr:
                    gender_missing_pred += 1
                else:
                    gender_correct += int(int(round(float(pred_attr["gender"]))) == int(round(float(gt_attr["gender"]))))
            if isinstance(gt_attr, dict) and "age" in gt_attr:
                age_total += 1
                if not isinstance(pred_attr, dict) or "age" not in pred_attr:
                    age_missing_pred += 1
                else:
                    err = abs(float(pred_attr["age"]) - float(gt_attr["age"]))
                    age_abs_err_sum += err
                    age_acc10 += int(err < 10.0)

        per_image.append(
            {
                "id": sid,
                "valid_prediction_json": bool(ok),
                "gt_box": len(gt_boxes),
                "pred_box": len(pred_boxes),
                "box_tp_iou_thr": len(matches),
                "box_fp_iou_thr": max(0, len(pred_boxes) - len(matched_p)),
                "box_fn_iou_thr": max(0, len(gt_boxes) - len(matched_g)),
                "pred_best_ious": best_ious,
                "box_matches": [
                    {"gt_index": int(gi), "pred_index": int(pi), "iou": float(iou)}
                    for gi, pi, iou in matches
                ],
                "pose_gt_valid_matched_box": img_pose_gt_valid,
                "pose_pck_correct": img_pose_kpt_correct,
                "pose_pck_total": img_pose_kpt_total,
            }
        )

    precision = tp_box / max(1, n_pred_box)
    recall = tp_box / max(1, n_gt_box)
    f1 = 0.0 if precision + recall <= 0 else 2 * precision * recall / (precision + recall)
    metrics = {
        "scheme": "scheme2_box_matched_incomplete_label_safe",
        "num_images": len(ids),
        "valid_json_rate": valid_json / max(1, len(ids)),
        "bbox_iou_thr": float(iou_thr),
        "bbox_num_gt": int(n_gt_box),
        "bbox_num_pred": int(n_pred_box),
        "bbox_tp": int(tp_box),
        "bbox_fp": int(max(0, n_pred_box - tp_box)),
        "bbox_fn": int(max(0, n_gt_box - tp_box)),
        "bbox_precision": float(precision),
        "bbox_recall": float(recall),
        "bbox_f1": float(f1),
        "bbox_AP50_image_mean": float(sum_ap50 / max(1, len(ids))),
        "bbox_AP5095_image_mean": float(sum_ap5095 / max(1, len(ids))),
        "pose_pck_thr": float(pck_thr),
        "pose_num_gt_valid_total": int(pose_gt_valid_total),
        "pose_num_gt_valid_on_matched_boxes": int(pose_gt_valid),
        "pose_num_gt_valid_unmatched_box": int(max(0, pose_gt_valid_total - pose_gt_valid)),
        "pose_pck_on_matched_boxes": float(pose_kpt_correct / max(1, pose_kpt_total)),
        "pose_pck_all_gt_visible": float(pose_kpt_correct / max(1, pose_visible_kpt_total_all_gt)),
        "pose_visible_kpt_correct": int(pose_kpt_correct),
        "pose_visible_kpt_total_on_matched_boxes": int(pose_kpt_total),
        "pose_visible_kpt_total_all_gt": int(pose_visible_kpt_total_all_gt),
        "pose_mean_oks_on_matched_boxes": float(sum(pose_oks_values) / max(1, len(pose_oks_values))),
        "attribute_gender_total": int(gender_total),
        "attribute_gender_acc": float(gender_correct / max(1, gender_total)),
        "attribute_gender_missing_pred": int(gender_missing_pred),
        "attribute_age_total": int(age_total),
        "attribute_age_mae": float(age_abs_err_sum / max(1, age_total - age_missing_pred)),
        "attribute_age_acc10": float(age_acc10 / max(1, age_total)),
        "attribute_age_missing_pred": int(age_missing_pred),
    }
    metrics["per_image"] = per_image
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scheme-2 eval for box+pose+attribute predictions.")
    parser.add_argument("--gt-jsonl", type=str, required=True)
    parser.add_argument("--pred-jsonl", type=str, required=True)
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--pck-thr", type=float, default=0.2)
    parser.add_argument("--hide-per-image", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_records_scheme2(
        gt_records=load_jsonl(args.gt_jsonl),
        pred_records=load_jsonl(args.pred_jsonl),
        iou_thr=float(args.iou_thr),
        pck_thr=float(args.pck_thr),
    )
    if bool(args.hide_per_image):
        metrics.pop("per_image", None)
    if args.output_json:
        dump_json(args.output_json, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
