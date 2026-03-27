import json
from typing import Any, Dict, Iterable, List, Optional


def xywh_to_xyxy(box: Iterable[float]) -> List[float]:
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def normalize_box_obj(box_obj: Any) -> Optional[Dict[str, List[float]]]:
    if isinstance(box_obj, dict):
        if "bbox_xyxy" in box_obj:
            box = box_obj["bbox_xyxy"]
        elif "bbox" in box_obj:
            box = box_obj["bbox"]
        elif {"x", "y", "w", "h"} <= set(box_obj.keys()):
            box = [box_obj["x"], box_obj["y"], box_obj["w"], box_obj["h"]]
            box = xywh_to_xyxy(box)
        else:
            return None
    elif isinstance(box_obj, (list, tuple)):
        if len(box_obj) == 4:
            box = list(box_obj)
        else:
            return None
    else:
        return None

    if len(box) != 4:
        return None
    return {"bbox_xyxy": [round(float(v), 2) for v in box]}


def build_box_target(sample: Dict[str, Any], max_instances: int = 100) -> str:
    boxes = []
    for box in sample.get("box", []):
        item = normalize_box_obj(box)
        if item is not None:
            boxes.append(item)
        if len(boxes) >= max_instances:
            break
    return json.dumps({"box": boxes}, ensure_ascii=False, separators=(",", ":"))
