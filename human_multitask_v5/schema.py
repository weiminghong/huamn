from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SCHEMA_VERSION = "human-multitask-v5"
TASKS = (
    "box", "pose", "segment", "parsing",
    "age", "gender", "attribute", "relation",
)

CANONICAL_JOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
    "head", "neck", "spine", "pelvis",
)

POSE_VISIBILITY_TO_CANONICAL = {
    "coco17": {0: 0, 1: 1, 2: 2},
    "mhp16": {0: 2, 1: 1, 2: 0},
}

POSE_SCHEMAS = {
    "coco17": CANONICAL_JOINTS[:17],
    "mhp16": (
        "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist",
        "right_hip", "right_knee", "right_ankle",
        "left_hip", "left_knee", "left_ankle",
        "head", "neck", "spine", "pelvis",
    ),
}

MASK_ENCODINGS = {"coco_rle", "polygons", "png", "label_png"}
PARSING_TAXONOMIES = {"mhp59", "cihp20"}


class SchemaError(ValueError):
    pass


def empty_label() -> Dict[str, Any]:
    return {"valid": False}


def canonical_pose(
    points: Sequence[Sequence[float]], schema: str
) -> Dict[str, Any]:
    if schema not in POSE_SCHEMAS:
        raise SchemaError(f"unsupported pose schema: {schema}")
    names = POSE_SCHEMAS[schema]
    if len(points) != len(names):
        raise SchemaError(
            f"{schema} requires {len(names)} joints, received {len(points)}"
        )
    by_name = {name: point for name, point in zip(names, points)}
    canonical: List[List[float]] = []
    valid: List[bool] = []
    for name in CANONICAL_JOINTS:
        if name not in by_name:
            canonical.append([0.0, 0.0, 0.0])
            valid.append(False)
            continue
        point = by_name[name]
        if len(point) != 3:
            raise SchemaError(f"joint {name} is not [x,y,visibility]")
        x, y, visibility = float(point[0]), float(point[1]), int(point[2])
        if visibility not in (0, 1, 2):
            raise SchemaError(f"joint {name} has invalid visibility={visibility}")
        canonical_visibility = POSE_VISIBILITY_TO_CANONICAL[schema][visibility]
        canonical.append([x, y, float(canonical_visibility)])
        valid.append(True)
    return {
        "valid": True,
        "source_schema": schema,
        "joint_names": list(CANONICAL_JOINTS),
        "points": canonical,
        "schema_valid": valid,
    }


def _require_box(box: Any, width: int, height: int) -> List[float]:
    if not isinstance(box, list) or len(box) != 4:
        raise SchemaError("bbox_xyxy must be a four-element list")
    values = [float(value) for value in box]
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise SchemaError(f"invalid bbox_xyxy={box} for image {(width, height)}")
    return values


def _validate_mask(label: Mapping[str, Any], *, parsing: bool) -> None:
    if not label.get("valid", False):
        return
    encoding = label.get("encoding")
    if not isinstance(encoding, Mapping):
        raise SchemaError("valid mask label requires an encoding object")
    if encoding.get("type") not in MASK_ENCODINGS:
        raise SchemaError(f"unsupported mask encoding: {encoding.get('type')}")
    if parsing and label.get("taxonomy") not in PARSING_TAXONOMIES:
        raise SchemaError(f"unsupported parsing taxonomy: {label.get('taxonomy')}")
    if encoding.get("type") in {"png", "label_png"}:
        path = encoding.get("path")
        if not isinstance(path, str) or not path:
            raise SchemaError("path-backed mask must have a non-empty path")


def available_tasks(record: Mapping[str, Any]) -> set[str]:
    tasks: set[str] = set()
    persons = record.get("persons", [])
    if persons:
        tasks.add("box")
    for person in persons:
        labels = person.get("labels", {})
        for task in (
            "pose", "segment", "parsing", "age", "gender", "attribute"
        ):
            if labels.get(task, {}).get("valid", False):
                tasks.add(task)
    if record.get("relations"):
        tasks.add("relation")
    return tasks


def validate_record(
    record: Mapping[str, Any], *, require_files: bool = False
) -> None:
    if record.get("schema") != SCHEMA_VERSION:
        raise SchemaError(f"record schema must be {SCHEMA_VERSION}")
    for key in ("uid", "dataset", "split", "image", "persons", "relations"):
        if key not in record:
            raise SchemaError(f"record is missing {key}")
    image = record["image"]
    if not isinstance(image, Mapping):
        raise SchemaError("image must be an object")
    width, height = int(image.get("width", 0)), int(image.get("height", 0))
    if width <= 0 or height <= 0:
        raise SchemaError("image dimensions must be positive")
    path = image.get("path")
    if not isinstance(path, str) or not path:
        raise SchemaError("image.path must be a non-empty string")
    if require_files and not Path(path).is_file():
        raise SchemaError(f"image is missing: {path}")

    persons = record["persons"]
    if not isinstance(persons, list):
        raise SchemaError("persons must be a list")
    source_ids = set()
    for expected_slot, person in enumerate(persons, start=1):
        if int(person.get("slot_id", -1)) != expected_slot:
            raise SchemaError(
                "person slots must be contiguous and deterministically ordered"
            )
        source_id = str(person.get("source_instance_id", ""))
        if not source_id or source_id in source_ids:
            raise SchemaError("source_instance_id must be non-empty and unique")
        source_ids.add(source_id)
        _require_box(person.get("bbox_xyxy"), width, height)
        labels = person.get("labels")
        if not isinstance(labels, Mapping):
            raise SchemaError("each person requires labels")
        label_tasks = (
            "pose", "segment", "parsing",
            "age", "gender", "attribute",
        )
        for task in label_tasks:
            if task not in labels or not isinstance(labels[task], Mapping):
                raise SchemaError(f"person label is missing task {task}")
        pose = labels["pose"]
        if pose.get("valid", False):
            if pose.get("joint_names") != list(CANONICAL_JOINTS):
                raise SchemaError("pose joint_names do not match canonical order")
            if len(pose.get("points", [])) != len(CANONICAL_JOINTS):
                raise SchemaError("canonical pose must contain 21 points")
            if len(pose.get("schema_valid", [])) != len(CANONICAL_JOINTS):
                raise SchemaError("pose schema_valid must contain 21 flags")
        _validate_mask(labels["segment"], parsing=False)
        _validate_mask(labels["parsing"], parsing=True)
        if labels["age"].get("valid", False):
            age = float(labels["age"].get("value"))
            if not 0 <= age <= 120:
                raise SchemaError("age must be in [0,120]")
        if labels["gender"].get("valid", False):
            if labels["gender"].get("value") not in (0, 1):
                raise SchemaError("gender must be 0/1 when valid")
        attribute = labels["attribute"]
        if attribute.get("valid", False):
            positive = set(attribute.get("positive", []))
            negative = set(attribute.get("negative", []))
            if positive & negative:
                raise SchemaError("attribute positives and negatives overlap")

    entity_ids = {
        str(entity.get("entity_id")) for entity in record.get("entities", [])
    }
    for edge in record["relations"]:
        if int(edge.get("subject_slot", -1)) not in range(1, len(persons) + 1):
            raise SchemaError("relation subject_slot does not reference a person")
        if not isinstance(edge.get("predicate"), str) or not edge["predicate"]:
            raise SchemaError("relation predicate must be non-empty")
        target = edge.get("object_ref", {})
        if not isinstance(target, Mapping):
            raise SchemaError("relation object_ref must be an object")
        kind = target.get("kind")
        if kind == "person":
            if int(target.get("slot_id", -1)) not in range(1, len(persons) + 1):
                raise SchemaError("relation object person does not exist")
        elif kind == "entity":
            if str(target.get("entity_id")) not in entity_ids:
                raise SchemaError("relation object entity does not exist")
        else:
            raise SchemaError(
                "relation object_ref.kind must be person or entity"
            )


@dataclass(frozen=True)
class ManifestSummary:
    records: int
    persons: int
    datasets: Dict[str, int]
    tasks: Dict[str, int]
    task_persons: Dict[str, int]


def summarize(records: Iterable[Mapping[str, Any]]) -> ManifestSummary:
    datasets: Counter[str] = Counter()
    tasks: Counter[str] = Counter()
    task_persons: Counter[str] = Counter()
    record_count = 0
    person_count = 0
    for record in records:
        validate_record(record)
        record_count += 1
        persons = record["persons"]
        person_count += len(persons)
        datasets[str(record["dataset"])] += 1
        for task in available_tasks(record):
            tasks[task] += 1
        for person in persons:
            labels = person["labels"]
            task_persons["box"] += 1
            for task in (
                "pose", "segment", "parsing",
                "age", "gender", "attribute",
            ):
                task_persons[task] += int(labels[task].get("valid", False))
        task_persons["relation"] += len(record["relations"])
    return ManifestSummary(
        records=record_count,
        persons=person_count,
        datasets=dict(sorted(datasets.items())),
        tasks=dict(sorted(tasks.items())),
        task_persons=dict(sorted(task_persons.items())),
    )
