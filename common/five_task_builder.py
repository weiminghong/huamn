import json
from typing import Any, Dict, List, Optional, Tuple


def _to_float(v: Any, precision: int = 2) -> float:
    try:
        return round(float(v), precision)
    except Exception:
        return 0.0


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _to_entity(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
        return stripped
    return str(value)


def _normalize_box(box_obj: Any) -> Optional[Dict[str, List[float]]]:
    if isinstance(box_obj, dict):
        box = box_obj.get("bbox_xyxy", box_obj.get("bbox", None))
    elif isinstance(box_obj, (list, tuple)):
        box = box_obj
    else:
        return None
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    return {"bbox_xyxy": [_to_float(x) for x in box]}


def _normalize_pose(pose_obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(pose_obj, list):
        pose_obj = {"keypoints": pose_obj}
    if not isinstance(pose_obj, dict):
        return None
    keypoints = pose_obj.get("keypoints", [])
    if not isinstance(keypoints, list):
        return None

    norm_kpts = []
    for kp in keypoints:
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        x = _to_float(kp[0])
        y = _to_float(kp[1])
        v = _to_float(kp[2]) if len(kp) > 2 else 2.0
        norm_kpts.append([x, y, v])
    return {"keypoints": norm_kpts}


def _normalize_attribute(attr_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(attr_obj, dict):
        return None
    norm_attr: Dict[str, Any] = {}
    for k, v in attr_obj.items():
        if isinstance(v, (int, float, bool, str)):
            norm_attr[str(k)] = v
        else:
            norm_attr[str(k)] = str(v)
    return norm_attr


def _normalize_relation(rel_obj: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(rel_obj, dict):
        return None
    subject = rel_obj.get("subject", rel_obj.get("subj", rel_obj.get("head", None)))
    predicate = rel_obj.get(
        "predicate", rel_obj.get("relation", rel_obj.get("rel", rel_obj.get("verb", None)))
    )
    obj = rel_obj.get("object", rel_obj.get("obj", rel_obj.get("tail", None)))
    if predicate is None:
        return None
    return {
        "subject": _to_entity(subject),
        "predicate": str(predicate),
        "object": _to_entity(obj),
    }


def _extract_mask_path(parsing_obj: Any) -> Optional[str]:
    if not isinstance(parsing_obj, dict):
        return None
    for key in ("mask_path", "mask", "seg_path", "path"):
        value = parsing_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_parsing_label(parsing_obj: Any) -> str:
    if not isinstance(parsing_obj, dict):
        return "part"
    for key in ("label", "part", "name", "class_name"):
        value = parsing_obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    labels = parsing_obj.get("labels", None)
    if isinstance(labels, list) and labels:
        return str(labels[0])
    return "part"


def normalize_five_task_sample(
    sample: Dict[str, Any],
    seg_token: str = "<SEG>",
    max_instances: int = 30,
) -> Tuple[Dict[str, Any], List[str]]:
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

    attrs = []
    attr_obj = sample.get("attribute", [])
    if isinstance(attr_obj, dict):
        attr_obj = [attr_obj]
    if isinstance(attr_obj, list):
        for item in attr_obj:
            norm_attr = _normalize_attribute(item)
            if norm_attr is not None:
                attrs.append(norm_attr)
            if len(attrs) >= max_instances:
                break

    relations = []
    rel_obj = sample.get("relation", [])
    if isinstance(rel_obj, dict):
        rel_obj = [rel_obj]
    if isinstance(rel_obj, list):
        for item in rel_obj:
            norm_rel = _normalize_relation(item)
            if norm_rel is not None:
                relations.append(norm_rel)
            if len(relations) >= max_instances:
                break

    parsing = []
    mask_paths: List[str] = []
    parsing_obj = sample.get("parsing", [])
    if isinstance(parsing_obj, dict):
        parsing_obj = [parsing_obj]
    if isinstance(parsing_obj, list):
        for item in parsing_obj:
            mask_path = _extract_mask_path(item)
            if not mask_path:
                continue
            parsing.append({"label": _extract_parsing_label(item), "mask_token": seg_token})
            mask_paths.append(mask_path)
            if len(parsing) >= max_instances:
                break

    payload = {
        "box": boxes,
        "pose": poses,
        "attribute": attrs,
        "relation": relations,
        "parsing": parsing,
    }
    return payload, mask_paths


def build_five_task_target_text(
    sample: Dict[str, Any],
    seg_token: str = "<SEG>",
    max_instances: int = 30,
) -> Tuple[str, List[str], Dict[str, Any]]:
    payload, mask_paths = normalize_five_task_sample(
        sample=sample,
        seg_token=seg_token,
        max_instances=max_instances,
    )
    target_text = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return target_text, mask_paths, payload
