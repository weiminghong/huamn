from __future__ import annotations

from typing import Any, Dict


SPECIAL_TOKENS_V1 = ["<box>", "<pose>", "<seg>"]

COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)

PROMPT_V1 = f"""You are a human-centric structured perception model.
Detect every clear visible real person exactly once and return one JSON object.

Return schema:
{{"persons":[{{"person_id":1,"box":"<box>","pose":"<pose>","seg":"<seg>","gender":0,"age":31}}]}}

Rules:
- Output JSON only. Do not output markdown, prose, comments, or chain-of-thought.
- Include every visible non-crowd person exactly once; never emit a second record for the same person.
- After the final unique person record, immediately close the JSON object and emit EOS.
- Order persons left-to-right by body-box center, then top-to-bottom.
- person_id values must be contiguous integers starting at 1.
- box must be exactly "<box>" for every record.
- pose is "<pose>" when pose is decoded, otherwise null.
- seg is "<seg>" when a person mask is decoded, otherwise null.
- COCO-17 pose order is: {COCO_KEYPOINT_ORDER}.
- gender is 0 for male, 1 for female, and -1 when unknown.
- age is apparent age in years and -1 when unknown.
- If no person is visible, return exactly {{"persons":[]}}.
"""

GUIDED_JSON_SCHEMA_V1: Dict[str, Any] = {
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
                    "box": {"type": "string", "enum": ["<box>"]},
                    "pose": {
                        "anyOf": [
                            {"type": "string", "enum": ["<pose>"]},
                            {"type": "null"},
                        ]
                    },
                    "seg": {
                        "anyOf": [
                            {"type": "string", "enum": ["<seg>"]},
                            {"type": "null"},
                        ]
                    },
                    "gender": {"type": "integer", "enum": [-1, 0, 1]},
                    "age": {"type": "integer", "minimum": -1, "maximum": 120},
                },
            },
        }
    },
}



def canonicalize_person_ids_v1(value: Dict[str, Any]) -> int:
    persons = value.get("persons", []) if isinstance(value, dict) else []
    if not isinstance(persons, list):
        return 0
    repairs = 0
    for expected, person in enumerate(persons, start=1):
        if not isinstance(person, dict):
            continue
        if person.get("person_id") != expected:
            person["person_id"] = expected
            repairs += 1
    return repairs

def validate_prediction_object_v1(value: Dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"persons"}:
        raise ValueError("prediction must be an object containing only persons")
    persons = value["persons"]
    if not isinstance(persons, list):
        raise ValueError("persons must be a list")
    required = {"person_id", "box", "pose", "seg", "gender", "age"}
    for index, person in enumerate(persons, start=1):
        if not isinstance(person, dict) or set(person) != required:
            raise ValueError(f"person {index} has invalid fields")
        if person["person_id"] != index:
            raise ValueError("person_id values must be contiguous and ordered")
        if person["box"] != "<box>":
            raise ValueError("box must be <box>")
        if person["pose"] not in ("<pose>", None):
            raise ValueError("pose must be <pose> or null")
        if person["seg"] not in ("<seg>", None):
            raise ValueError("seg must be <seg> or null")
        if person["gender"] not in (-1, 0, 1):
            raise ValueError("gender is outside the schema")
        if not isinstance(person["age"], int) or not -1 <= person["age"] <= 120:
            raise ValueError("age is outside the schema")

