import json

import pytest

from human_multitask_v5.protocol import build_target, packet_tasks
from human_multitask_v5.schema import (
    CANONICAL_JOINTS,
    SchemaError,
    canonical_pose,
    empty_label,
    validate_record,
)


def record():
    pose = canonical_pose([[10 + i, 20 + i, 2] for i in range(17)], "coco17")
    person = {
        "slot_id": 1,
        "source_instance_id": "7",
        "bbox_xyxy": [1, 2, 51, 92],
        "area": 4500,
        "labels": {
            "pose": pose,
            "segment": {
                "valid": True,
                "encoding": {"type": "polygons", "data": [[1, 2, 51, 2, 51, 92]]},
            },
            "parsing": empty_label(),
            "age": {"valid": True, "value": 30},
            "gender": {"valid": True, "value": 1},
            "attribute": {
                "valid": True,
                "positive": ["red shirt"],
                "negative": ["hat"],
            },
        },
    }
    return {
        "schema": "human-multitask-v5",
        "uid": "coco:train:1",
        "dataset": "coco_unihuman",
        "split": "train",
        "image": {
            "path": "/tmp/not-required.jpg",
            "width": 100,
            "height": 100,
            "source_image_id": 1,
            "coco_id": 1,
        },
        "persons": [person],
        "entities": [],
        "relations": [],
    }


def test_pose_union_and_protocol():
    value = record()
    validate_record(value)
    pose = value["persons"][0]["labels"]["pose"]
    assert len(pose["points"]) == 21
    assert pose["schema_valid"][:17] == [True] * 17
    assert pose["schema_valid"][17:] == [False] * 4
    output = json.loads(build_target(value, "pose"))
    assert output["persons"][0]["slot"] == "<human>"
    assert len(output["persons"][0]["pose"]["joints"]) == len(CANONICAL_JOINTS)
    assert output["persons"][0]["pose"]["joints"][-1]["value"] is None
    assert set(packet_tasks(value)) >= {
        "box", "pose", "segment", "age", "gender", "attribute", "all"
    }


def test_unannotated_is_not_negative():
    value = record()
    output = json.loads(build_target(value, "parsing"))
    assert len(output["persons"]) == 1
    assert "parsing" not in output["persons"][0]


def test_bad_relation_owner_fails():
    value = record()
    value["relations"] = [
        {
            "subject_slot": 2,
            "predicate": "holding",
            "object_ref": {"kind": "person", "slot_id": 1},
        }
    ]
    with pytest.raises(SchemaError):
        validate_record(value)


def test_mhp_visibility_is_converted_to_coco_semantics():
    points = [[10 + i, 20 + i, i % 3] for i in range(16)]
    pose = canonical_pose(points, "mhp16")
    source_names = [
        name for name, valid in zip(pose["joint_names"], pose["schema_valid"])
        if valid
    ]
    source_values = [
        int(point[2]) for point, valid in zip(pose["points"], pose["schema_valid"])
        if valid
    ]
    assert source_names[0] == "left_shoulder"
    # MHP source left shoulder is index 3 with state visible=0 -> canonical 2.
    assert source_values[0] == 2
    # MHP source right shoulder is index 0 with state visible=0 -> canonical 2.
    right_index = source_names.index("right_shoulder")
    assert source_values[right_index] == 2
