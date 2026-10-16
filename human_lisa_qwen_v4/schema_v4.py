from __future__ import annotations

from typing import Any, Dict


POSE_TOKEN = "<pose>"
SEG_TOKEN = "<seg>"
SPECIAL_TOKENS_V4 = [POSE_TOKEN, SEG_TOKEN]
BOX_SCALE = 1000
THINKING_MODE_V4 = "disabled"


def apply_chat_template_v4(processor: Any, messages: list[dict]) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return processor.apply_chat_template(messages, **kwargs)


PROMPT_V4 = """Detect every visible non-crowd real person exactly once.
Return one JSON object and nothing else.
Use this exact schema:
{"persons":[{"person_id":1,"box":[x1,y1,x2,y2],"pose":"<pose>","seg":"<seg>","gender":0,"age":31}]}
Rules:
- Order people left-to-right by body-box center, then top-to-bottom.
- person_id must be contiguous starting at 1.
- box is the full visible body box as four integers from 0 to 1000.
- box order is x1,y1,x2,y2 with x1<x2 and y1<y2.
- pose must be exactly "<pose>"; its decoder predicts COCO-17 keypoints.
- seg must be exactly "<seg>"; its decoder predicts the person mask.
- gender is 0 for male, 1 for female, and -1 only when genuinely unknown.
- age is apparent age in years from 0 to 120, or -1 when genuinely unknown.
- Do not repeat a person. Close the JSON immediately after the last person.
- If no person is visible, return exactly {"persons":[]}.
"""

GUIDED_JSON_SCHEMA_V4: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["persons"],
    "properties": {
        "persons": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "person_id",
                    "box",
                    "pose",
                    "seg",
                    "gender",
                    "age",
                ],
                "properties": {
                    "person_id": {"type": "integer", "minimum": 1},
                    "box": {
                        "type": "array",
                        "prefixItems": [
                            {"type": "integer", "minimum": 0, "maximum": BOX_SCALE},
                            {"type": "integer", "minimum": 0, "maximum": BOX_SCALE},
                            {"type": "integer", "minimum": 0, "maximum": BOX_SCALE},
                            {"type": "integer", "minimum": 0, "maximum": BOX_SCALE},
                        ],
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "pose": {"type": "string", "enum": [POSE_TOKEN]},
                    "seg": {"type": "string", "enum": [SEG_TOKEN]},
                    "gender": {"type": "integer", "enum": [-1, 0, 1]},
                    "age": {"type": "integer", "minimum": -1, "maximum": 120},
                },
            },
        }
    },
}


def canonicalize_person_ids_v4(value: Dict[str, Any]) -> int:
    persons = value.get("persons") if isinstance(value, dict) else None
    if not isinstance(persons, list):
        return 0
    repairs = 0
    for expected_id, person in enumerate(persons, start=1):
        if not isinstance(person, dict):
            continue
        if person.get("person_id") != expected_id:
            person["person_id"] = expected_id
            repairs += 1
    return repairs


def validate_prediction_object_v4(value: Dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"persons"}:
        raise ValueError("prediction must contain only the persons field")
    persons = value["persons"]
    if not isinstance(persons, list):
        raise ValueError("persons must be a list")
    required = {"person_id", "box", "pose", "seg", "gender", "age"}
    for expected_id, person in enumerate(persons, start=1):
        if not isinstance(person, dict) or set(person) != required:
            raise ValueError(f"person {expected_id} has invalid fields")
        if person["person_id"] != expected_id:
            raise ValueError("person_id values must be contiguous and ordered")
        box = person["box"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(type(value) is not int for value in box)
            or any(value < 0 or value > BOX_SCALE for value in box)
            or not box[0] < box[2]
            or not box[1] < box[3]
        ):
            raise ValueError("box is outside the v4 coordinate contract")
        if person["pose"] != POSE_TOKEN or person["seg"] != SEG_TOKEN:
            raise ValueError("pose/seg routing tokens are invalid")
        if person["gender"] not in (-1, 0, 1):
            raise ValueError("gender is outside the schema")
        if type(person["age"]) is not int or not -1 <= person["age"] <= 120:
            raise ValueError("age is outside the schema")
