from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from .schema import CANONICAL_JOINTS, available_tasks


POSE_TOKEN = "<pose>"
SEG_TOKEN = "<seg>"
PARSE_TOKEN = "<parse>"
HUMAN_TOKEN = "<human>"
SPECIAL_TOKENS = (HUMAN_TOKEN, POSE_TOKEN, SEG_TOKEN, PARSE_TOKEN)


@dataclass(frozen=True)
class PacketRef:
    record_index: int
    dataset: str
    task: str
    joint: bool = False


def quantize_box(
    box: Sequence[float], width: int, height: int
) -> List[int]:
    scales = (width, height, width, height)
    result = [
        round(max(0.0, min(1.0, float(value) / scale)) * 1000)
        for value, scale in zip(box, scales)
    ]
    result[2] = max(result[2], min(1000, result[0] + 1))
    result[3] = max(result[3], min(1000, result[1] + 1))
    return result


def pose_payload(
    person: Mapping[str, Any], width: int, height: int
) -> Dict[str, Any]:
    pose = person["labels"]["pose"]
    joints = []
    for name, point, schema_valid in zip(
        CANONICAL_JOINTS, pose["points"], pose["schema_valid"]
    ):
        if not schema_valid:
            joints.append({"name": name, "value": None})
            continue
        x, y, visibility = point
        value = [
            round(max(0.0, min(1.0, float(x) / width)) * 1000),
            round(max(0.0, min(1.0, float(y) / height)) * 1000),
            int(visibility),
        ]
        joints.append({"name": name, "value": value})
    return {
        "box": quantize_box(person["bbox_xyxy"], width, height),
        "joints": joints,
    }


def build_target(record: Mapping[str, Any], task: str) -> str:
    width = int(record["image"]["width"])
    height = int(record["image"]["height"])
    persons = []
    for person in record["persons"]:
        item: Dict[str, Any] = {
            "person_id": int(person["slot_id"]),
            "slot": HUMAN_TOKEN,
            "box": quantize_box(person["bbox_xyxy"], width, height),
        }
        labels = person["labels"]
        if task in ("all", "pose") and labels["pose"].get("valid", False):
            item["pose"] = pose_payload(person, width, height)
        if task in ("all", "segment") and labels["segment"].get("valid", False):
            item["segment"] = SEG_TOKEN
        if task in ("all", "parsing") and labels["parsing"].get("valid", False):
            item["parsing"] = PARSE_TOKEN
        if task in ("all", "age") and labels["age"].get("valid", False):
            item["age"] = labels["age"]["value"]
        if task in ("all", "gender") and labels["gender"].get("valid", False):
            item["gender"] = labels["gender"]["value"]
        if task in ("all", "attribute") and labels["attribute"].get("valid", False):
            item["attributes"] = list(labels["attribute"].get("positive", []))
        persons.append(item)

    payload: Dict[str, Any] = {"persons": persons}
    if task in ("all", "relation"):
        payload["relations"] = [
            {
                "subject_person_id": int(edge["subject_slot"]),
                "predicate": edge["predicate"],
                "object": edge["object_ref"],
            }
            for edge in record["relations"]
        ]
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=False
    )


def task_prompt(task: str) -> str:
    common = (
        "Identify every relevant visible person exactly once. "
        "Use contiguous person_id values ordered left-to-right, then top-to-bottom. "
        "Coordinates are integers from 0 to 1000. Return JSON only. "
    )
    prompts = {
        "box": "Return each person's body box.",
        "pose": "Return each labeled person's box and 21 named pose entries; use null only for joints absent from that dataset schema.",
        "segment": f"Return each labeled person's box and segment field exactly {SEG_TOKEN}.",
        "parsing": f"Return each labeled person's box and parsing field exactly {PARSE_TOKEN}.",
        "age": "Return each labeled person's box and apparent age.",
        "gender": "Return each labeled person's box and dataset gender label.",
        "attribute": "Return each labeled person's box and explicitly positive visual attributes only.",
        "relation": "Return person-grounded relation triples and retain the referenced object identity.",
        "all": "Return every available labeled field; omitted fields are unannotated, not negative.",
    }
    if task not in prompts:
        raise KeyError(task)
    return common + prompts[task]


def packet_tasks(
    record: Mapping[str, Any], *, allow_joint: bool = True
) -> List[str]:
    tasks = sorted(available_tasks(record))
    if allow_joint and len(tasks) >= 2:
        tasks.append("all")
    return tasks
