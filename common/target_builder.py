import json
from typing import Any, Dict, List, Optional


def _to_float(v: Any, precision: int = 2) -> float:
    try:
        return round(float(v), precision)
    except Exception:
        return 0.0


def _normalize_box(box_obj: Any) -> Optional[Dict[str, List[float]]]:
    if isinstance(box_obj, dict):
        if "bbox_xyxy" in box_obj and isinstance(box_obj["bbox_xyxy"], (list, tuple)):
            box = box_obj["bbox_xyxy"]
        elif "bbox" in box_obj and isinstance(box_obj["bbox"], (list, tuple)):
            box = box_obj["bbox"]
        else:
            return None
    elif isinstance(box_obj, (list, tuple)):
        box = box_obj
    else:
        return None

    if len(box) != 4:
        return None
    return {"bbox_xyxy": [_to_float(x) for x in box]}


def _normalize_pose(pose_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(pose_obj, dict):
        if isinstance(pose_obj, list):
            pose_obj = {"keypoints": pose_obj}
        else:
            return None

    keypoints = pose_obj.get("keypoints", [])
    if not isinstance(keypoints, list):
        return None

    norm_kpts = []
    for kp in keypoints:
        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
            x = _to_float(kp[0])
            y = _to_float(kp[1])
            v = _to_float(kp[2]) if len(kp) > 2 else 2.0
            norm_kpts.append([x, y, v])

    return {"keypoints": norm_kpts}


def _normalize_parsing(parsing_obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(parsing_obj, dict):
        labels = parsing_obj.get("labels", [])
        segments = parsing_obj.get("segments", [])
    elif isinstance(parsing_obj, list):
        labels = parsing_obj
        segments = []
    else:
        return None

    norm_labels = []
    if isinstance(labels, list):
        for v in labels:
            if isinstance(v, (int, float, str)):
                norm_labels.append(v)

    norm_segments = segments if isinstance(segments, list) else []
    return {"labels": norm_labels, "segments": norm_segments}


def build_stage1_target(sample: Dict[str, Any], max_instances: int = 30) -> str:
    boxes = []
    for box in sample.get("box", []):
        item = _normalize_box(box)
        if item is not None:
            boxes.append(item)
        if len(boxes) >= max_instances:
            break

    poses = []
    for pose in sample.get("pose", []):
        item = _normalize_pose(pose)
        if item is not None:
            poses.append(item)
        if len(poses) >= max_instances:
            break

    parsings = []
    for parsing in sample.get("parsing", []):
        item = _normalize_parsing(parsing)
        if item is not None:
            parsings.append(item)
        if len(parsings) >= max_instances:
            break

    payload = {"box": boxes, "pose": poses, "parsing": parsings}
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def parse_json_object_from_text(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or start >= end:
        return None

    snippet = text[start : end + 1]
    try:
        obj = json.loads(snippet)
        if isinstance(obj, dict):
            return obj
    except Exception:
        return None

    return None
