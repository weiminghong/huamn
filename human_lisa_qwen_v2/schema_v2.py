from __future__ import annotations

from typing import Any, Dict


SPECIAL_TOKENS_V2 = ["<person>", "<pose>", "<seg>"]

COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)

PROMPT_V2 = f"""You are a human-centric structured perception model.
Find every visible non-crowd real person exactly once and return one JSON object.

Return schema:
{{"persons":[{{"person_id":1,"person":"<person>","pose":"<pose>","seg":"<seg>","bbox":[120,80,640,980],"gender":0,"age":31}}]}}

Rules:
- Output JSON only. Do not output markdown, prose, comments, or chain-of-thought.
- Include every visible person exactly once, including small and partially occluded people.
- Order people left-to-right by full-body bounding-box center, then top-to-bottom.
- person_id values must be contiguous integers starting at 1.
- person, pose, and seg must be exactly "<person>", "<pose>", and "<seg>" in every person record.
- bbox is [x1,y1,x2,y2] in integer coordinates normalized to 0..1000; require x1 < x2 and y1 < y2.
- The pose decoder uses COCO-17 order: {COCO_KEYPOINT_ORDER}.
- gender is 0 for male, 1 for female, and -1 when genuinely unknown.
- age is apparent age in years from 0 to 120, and -1 when genuinely unknown.
- After the final unique person, immediately close the JSON object and emit EOS.
- If no person is visible, return exactly {{"persons":[]}}.
"""

GUIDED_JSON_SCHEMA_V2: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["persons"],
    "properties": {
        "persons": {
            "type": "array",
            "maxItems": 25,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "person_id",
                    "person",
                    "pose",
                    "seg",
                    "bbox",
                    "gender",
                    "age",
                ],
                "properties": {
                    "person_id": {"type": "integer", "minimum": 1},
                    "person": {"type": "string", "enum": ["<person>"]},
                    "pose": {"type": "string", "enum": ["<pose>"]},
                    "seg": {"type": "string", "enum": ["<seg>"]},
                    "bbox": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0, "maximum": 1000},
                        "minItems": 4,
                        "maxItems": 4,
                    },
                    "gender": {"type": "integer", "enum": [-1, 0, 1]},
                    "age": {"type": "integer", "minimum": -1, "maximum": 120},
                },
            },
        }
    },
}


def canonicalize_person_ids_v2(value: Dict[str, Any]) -> int:
    persons = value.get("persons", []) if isinstance(value, dict) else []
    if not isinstance(persons, list):
        return 0
    repairs = 0
    for expected, person in enumerate(persons, start=1):
        if isinstance(person, dict) and person.get("person_id") != expected:
            person["person_id"] = expected
            repairs += 1
    return repairs


def validate_prediction_object_v2(value: Dict[str, Any]) -> None:
    if not isinstance(value, dict) or set(value) != {"persons"}:
        raise ValueError("prediction must be an object containing only persons")
    persons = value["persons"]
    if not isinstance(persons, list):
        raise ValueError("persons must be a list")
    required = {"person_id", "person", "pose", "seg", "bbox", "gender", "age"}
    for index, person in enumerate(persons, start=1):
        if not isinstance(person, dict) or set(person) != required:
            raise ValueError(f"person {index} has invalid fields")
        if person["person_id"] != index:
            raise ValueError("person_id values must be contiguous and ordered")
        if person["person"] != "<person>" or person["pose"] != "<pose>" or person["seg"] != "<seg>":
            raise ValueError("person/pose/seg anchors must use their exact special tokens")
        box = person["bbox"]
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(not isinstance(value, int) or not 0 <= value <= 1000 for value in box)
            or not box[0] < box[2]
            or not box[1] < box[3]
        ):
            raise ValueError("bbox must be a valid normalized integer xyxy box")
        if person["gender"] not in (-1, 0, 1):
            raise ValueError("gender is outside the schema")
        if not isinstance(person["age"], int) or not -1 <= person["age"] <= 120:
            raise ValueError("age is outside the schema")
