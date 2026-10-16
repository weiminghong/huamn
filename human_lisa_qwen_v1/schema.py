from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SPECIAL_TOKENS = ["<box>", "<pose>", "<seg>"]
BOX_TOKEN = "<box>"
POSE_TOKEN = "<pose>"
SEG_TOKEN = "<seg>"

COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)

PROMPT = f'''You are a human-centric structured perception model.
Detect every clear visible real person exactly once and return one JSON object.

Return schema:
{{"persons":[{{"person_id":1,"box":"<box>","pose":"<pose>","seg":null,"gender":-1,"age":-1,"score":1.0}}]}}

Rules:
- Output JSON only. No markdown, prose, comments, or chain-of-thought.
- persons are ordered left-to-right by body box center; break ties top-to-bottom.
- "box" must be exactly "<box>" for each emitted person. The box head decodes coordinates from that token.
- "pose" is "<pose>" when a person pose should be decoded; otherwise null. The pose head decodes COCO-17 keypoints from that token.
- "seg" is "<seg>" only when segmentation/parsing supervision is available; otherwise null. The seg token is a LISA-style hook for a mask decoder.
- COCO-17 pose order is: {COCO_KEYPOINT_ORDER}.
- gender: 0 male, 1 female, -1 unknown.
- age: apparent age in years, or -1 unknown.
- score: detection confidence in [0,1]. Use 1.0 when score calibration is unavailable.
- If no clear visible real person exists, output exactly {{"persons":[]}}.
'''

GUIDED_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["persons"],
    "properties": {
        "persons": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["person_id", "box", "pose", "seg", "gender", "age", "score"],
                "properties": {
                    "person_id": {"type": "integer", "minimum": 1, "maximum": 200},
                    "box": {"type": "string", "enum": [BOX_TOKEN]},
                    "pose": {"anyOf": [{"type": "string", "enum": [POSE_TOKEN]}, {"type": "null"}]},
                    "seg": {"anyOf": [{"type": "string", "enum": [SEG_TOKEN]}, {"type": "null"}]},
                    "gender": {"type": "integer", "enum": [-1, 0, 1]},
                    "age": {"type": "integer", "minimum": -1, "maximum": 120},
                    "score": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        }
    },
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resolve_image_path(record: Dict[str, Any], image_root: str = "") -> str:
    image = str(record.get("image_abs", record.get("image", "")))
    p = Path(image)
    if p.is_absolute():
        return str(p)
    if image_root:
        return str(Path(image_root) / image)
    return image


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        y = float(x)
    except Exception:
        return default
    if not math.isfinite(y):
        return default
    return y


def norm_box_xyxy(box: Iterable[Any], width: int, height: int) -> List[float]:
    vals = list(box)[:4]
    x1, y1, x2, y2 = [_safe_float(v) for v in vals]
    width = max(int(width), 1)
    height = max(int(height), 1)
    x1 = max(0.0, min(x1, width - 1.0)) / float(width)
    x2 = max(0.0, min(x2, width - 1.0)) / float(width)
    y1 = max(0.0, min(y1, height - 1.0)) / float(height)
    y2 = max(0.0, min(y2, height - 1.0)) / float(height)
    xa, xb = sorted([x1, x2])
    ya, yb = sorted([y1, y2])
    return [xa, ya, max(xb, xa + 1e-4), max(yb, ya + 1e-4)]


def denorm_box_xyxy(box: Iterable[Any], width: int, height: int) -> List[float]:
    x1, y1, x2, y2 = [_safe_float(v) for v in list(box)[:4]]
    return [x1 * width, y1 * height, x2 * width, y2 * height]


def attrs_by_pid(record: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {a.get("person_id"): a for a in record.get("attribute", []) if isinstance(a, dict)}


def poses_by_pid(record: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    return {p.get("person_id"): p for p in record.get("pose", []) if isinstance(p, dict)}


def segs_by_pid(record: Dict[str, Any]) -> Dict[Any, Dict[str, Any]]:
    for key in ("seg", "segmentation", "mask", "parsing"):
        items = record.get(key)
        if isinstance(items, list):
            return {s.get("person_id"): s for s in items if isinstance(s, dict)}
    return {}


def norm_pose_17(pose: Optional[Dict[str, Any]], width: int, height: int) -> Optional[List[List[float]]]:
    if not isinstance(pose, dict):
        return None
    raw = pose.get("keypoints", pose.get("keypoints_2d", []))
    if not isinstance(raw, list) or len(raw) < 17:
        return None
    out: List[List[float]] = []
    for kp in raw[:17]:
        if not isinstance(kp, (list, tuple)) or len(kp) < 3:
            out.append([0.0, 0.0, 0.0])
            continue
        v = int(round(_safe_float(kp[2], 0.0)))
        if v <= 0:
            out.append([0.0, 0.0, 0.0])
        else:
            x = max(0.0, min(_safe_float(kp[0]), width - 1.0)) / float(max(width, 1))
            y = max(0.0, min(_safe_float(kp[1]), height - 1.0)) / float(max(height, 1))
            out.append([x, y, 2.0 if v > 1 else 1.0])
    return out


def denorm_pose_17(pose: List[List[float]], width: int, height: int) -> List[List[float]]:
    out: List[List[float]] = []
    for kp in pose[:17]:
        x, y, v = kp
        vv = int(round(_safe_float(v)))
        if vv <= 0:
            out.append([0.0, 0.0, 0.0])
        else:
            out.append([_safe_float(x) * width, _safe_float(y) * height, float(vv)])
    while len(out) < 17:
        out.append([0.0, 0.0, 0.0])
    return out[:17]


def sorted_person_items(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    width = int(record.get("width", 1))
    height = int(record.get("height", 1))
    items = []
    for i, box_item in enumerate(record.get("box", []) or []):
        if not isinstance(box_item, dict):
            continue
        bbox = box_item.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = bbox[:4]
        cx = (_safe_float(x1) + _safe_float(x2)) * 0.5
        cy = (_safe_float(y1) + _safe_float(y2)) * 0.5
        item = dict(box_item)
        item["_sort_key"] = (cx, cy, i)
        item["_norm_box"] = norm_box_xyxy(bbox, width, height)
        items.append(item)
    items.sort(key=lambda x: x["_sort_key"])
    return items


def build_target(record: Dict[str, Any], include_seg_if_available: bool = True) -> Tuple[str, Dict[str, Any]]:
    width = int(record.get("width", 1))
    height = int(record.get("height", 1))
    attr_map = attrs_by_pid(record)
    pose_map = poses_by_pid(record)
    seg_map = segs_by_pid(record)

    persons_json: List[Dict[str, Any]] = []
    box_targets: List[List[float]] = []
    pose_targets: List[List[List[float]]] = []
    seg_targets: List[Any] = []

    for out_pid, box_item in enumerate(sorted_person_items(record), start=1):
        pid = box_item.get("person_id", out_pid)
        attr = attr_map.get(pid, {})
        pose = norm_pose_17(pose_map.get(pid), width, height)
        has_seg = include_seg_if_available and pid in seg_map
        gender = int(attr.get("gender", -1)) if isinstance(attr, dict) else -1
        if gender not in (-1, 0, 1):
            gender = -1
        age = -1
        if isinstance(attr, dict) and "age" in attr:
            age = int(round(_safe_float(attr.get("age"), -1.0)))
            if age < 0 or age > 120:
                age = -1
        persons_json.append({
            "person_id": out_pid,
            "box": BOX_TOKEN,
            "pose": POSE_TOKEN if pose is not None else None,
            "seg": SEG_TOKEN if has_seg else None,
            "gender": gender,
            "age": age,
            "score": 1.0,
        })
        box_targets.append(box_item["_norm_box"])
        if pose is not None:
            pose_targets.append(pose)
        if has_seg:
            seg_targets.append(seg_map[pid])
    target = json.dumps({"persons": persons_json}, ensure_ascii=False, separators=(",", ":"))
    aux = {"box_targets": box_targets, "pose_targets": pose_targets, "seg_targets": seg_targets}
    return target, aux


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_plain_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_prediction_object(obj: Dict[str, Any]) -> None:
    if not isinstance(obj, dict):
        raise ValueError("prediction must be a JSON object")
    if set(obj.keys()) != {"persons"}:
        raise ValueError("prediction object must contain only persons")
    persons = obj.get("persons")
    if not isinstance(persons, list):
        raise ValueError("persons must be a list")
    required = {"person_id", "box", "pose", "seg", "gender", "age", "score"}
    for idx, person in enumerate(persons):
        if not isinstance(person, dict):
            raise ValueError(f"persons[{idx}] must be an object")
        if set(person.keys()) != required:
            raise ValueError(f"persons[{idx}] fields do not match required schema")
        person_id = person.get("person_id")
        if not _is_plain_int(person_id) or not (1 <= int(person_id) <= 200):
            raise ValueError(f"persons[{idx}].person_id out of range")
        if person.get("box") != BOX_TOKEN:
            raise ValueError(f"persons[{idx}].box must be {BOX_TOKEN}")
        if person.get("pose") not in (POSE_TOKEN, None):
            raise ValueError(f"persons[{idx}].pose must be {POSE_TOKEN} or null")
        if person.get("seg") not in (SEG_TOKEN, None):
            raise ValueError(f"persons[{idx}].seg must be {SEG_TOKEN} or null")
        gender = person.get("gender")
        if not _is_plain_int(gender) or int(gender) not in (-1, 0, 1):
            raise ValueError(f"persons[{idx}].gender must be -1, 0, or 1")
        age = person.get("age")
        if not _is_plain_int(age) or not (-1 <= int(age) <= 120):
            raise ValueError(f"persons[{idx}].age out of range")
        score = person.get("score")
        if not _is_plain_number(score) or not (0.0 <= float(score) <= 1.0):
            raise ValueError(f"persons[{idx}].score out of range")


def parse_prediction_json(text: str) -> Dict[str, Any]:
    obj = json.loads(text)
    validate_prediction_object(obj)
    return obj
