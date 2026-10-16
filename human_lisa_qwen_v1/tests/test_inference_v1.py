import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.inference_v1 import (
    _pose_decoder_type_from_checkpoint,
    _target_token_probability_from_hidden,
    assemble_coco_records,
    combine_detection_score,
    semantic_health,
    suppress_near_duplicate_person_records,
    summarize_inference_audit,
)
from human_lisa_qwen_v1.schema_v1 import (
    canonicalize_person_ids_v1,
    validate_prediction_object_v1,
)


def test_detection_score_is_geometric_mean_and_bounded():
    assert combine_detection_score(0.25, 1.0) == pytest.approx(0.5)
    assert combine_detection_score(1.0, 0.25) == pytest.approx(0.5)
    assert combine_detection_score(-1.0, 2.0) == pytest.approx(0.0)


def test_semantic_health_detects_count_mismatch_and_duplicate_boxes():
    obj = {
        "persons": [
            {
                "person_id": 1,
                "box": "<box>",
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 0,
                "age": 20,
            },
            {
                "person_id": 2,
                "box": "<box>",
                "pose": None,
                "seg": None,
                "gender": -1,
                "age": -1,
            },
        ]
    }
    healthy = semantic_health(
        obj,
        boxes=torch.tensor([[0.1, 0.1, 0.3, 0.4], [0.6, 0.1, 0.8, 0.4]]),
        pose_count=1,
        seg_count=1,
        eos_seen=True,
        raw_token_count=100,
    )
    assert healthy["ok"] is True
    duplicate = semantic_health(
        obj,
        boxes=torch.tensor([[0.1, 0.1, 0.3, 0.4], [0.1, 0.1, 0.3, 0.4]]),
        pose_count=0,
        seg_count=1,
        eos_seen=False,
        raw_token_count=5000,
    )
    assert duplicate["ok"] is False
    assert duplicate["duplicate_pairs"] == 1
    assert "pose_count_mismatch" in duplicate["errors"]
    assert "missing_eos" in duplicate["errors"]


def test_near_duplicate_record_nms_preserves_atomic_person_binding():
    records = [
        {
            "person_id": 1,
            "bbox": [0.0, 0.0, 100.0, 100.0],
            "score": 0.9,
            "keypoints": [1.0] * 51,
            "segmentation": {"size": [1, 1], "counts": "1"},
            "gender": 0,
            "age": 20.0,
        },
        {
            "person_id": 2,
            "bbox": [1.0, 1.0, 100.0, 100.0],
            "score": 0.8,
            "keypoints": [2.0] * 51,
            "segmentation": {"size": [1, 1], "counts": "1"},
            "gender": 1,
            "age": 30.0,
        },
        {
            "person_id": 3,
            "bbox": [160.0, 0.0, 80.0, 100.0],
            "score": 0.7,
            "gender": 0,
            "age": 40.0,
        },
    ]
    kept, audit = suppress_near_duplicate_person_records(records, iou_threshold=0.9)
    assert [record["person_id"] for record in kept] == [1, 3]
    assert kept[0]["keypoints"] == [1.0] * 51
    assert audit["raw_records"] == 3
    assert audit["kept_records"] == 2
    assert audit["suppressed_records"][0]["person_id"] == 2
    assert audit["suppressed_records"][0]["kept_person_id"] == 1


def test_assemble_coco_records_keeps_person_binding():
    obj = {
        "persons": [
            {
                "person_id": 1,
                "box": "<box>",
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 0,
                "age": 25,
            }
        ]
    }
    pose = torch.zeros((1, 17, 2))
    pose[..., 0] = 0.5
    pose[..., 1] = 0.25
    visibility = torch.zeros((1, 17, 3))
    visibility[..., 2] = 10
    mask_logits = torch.full((1, 8, 8), -10.0)
    mask_logits[:, 2:6, 2:6] = 10.0
    records = assemble_coco_records(
        image_id=9,
        width=200,
        height=100,
        obj=obj,
        boxes=torch.tensor([[0.1, 0.2, 0.6, 0.8]]),
        quality=torch.tensor([0.81]),
        token_probabilities=torch.tensor([0.25]),
        pose_xy=pose,
        pose_visibility_logits=visibility,
        mask_logits=mask_logits,
    )
    assert len(records) == 1
    record = records[0]
    assert record["image_id"] == 9
    assert record["bbox"] == pytest.approx([20.0, 20.0, 100.0, 60.0])
    assert record["score"] == pytest.approx(0.45)
    assert len(record["keypoints"]) == 51
    assert record["keypoints"][:3] == pytest.approx([100.0, 25.0, 2.0])
    assert record["gender"] == 0
    assert record["age"] == 25
    assert isinstance(record["segmentation"]["counts"], str)



def test_target_token_probability_uses_only_preceding_hidden_rows():
    output_head = torch.nn.Linear(3, 5, bias=False)
    with torch.no_grad():
        output_head.weight.copy_(
            torch.tensor(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                    [-1.0, 0.0, 1.0],
                ]
            )
        )

    class FakeModel:
        def get_output_embeddings(self):
            return output_head

    hidden = torch.tensor(
        [[[0.1, 0.2, 0.3], [0.4, 0.2, -0.1], [0.0, 0.5, 0.1]]]
    )
    probabilities = _target_token_probability_from_hidden(
        FakeModel(),
        hidden,
        positions=[1, 3],
        token_id=3,
    )
    expected_logits = output_head(hidden[0, torch.tensor([0, 2])])
    expected = expected_logits.softmax(dim=-1)[:, 3]
    torch.testing.assert_close(probabilities, expected)
    with pytest.raises(RuntimeError, match="no preceding"):
        _target_token_probability_from_hidden(
            FakeModel(), hidden, positions=[0], token_id=3
        )


def test_pose_export_preserves_xy_when_visibility_predicts_zero():
    obj = {
        "persons": [
            {
                "person_id": 1,
                "box": "<box>",
                "pose": "<pose>",
                "seg": None,
                "gender": -1,
                "age": -1,
            }
        ]
    }
    pose = torch.full((1, 17, 2), 0.5)
    visibility = torch.zeros((1, 17, 3))
    visibility[..., 0] = 10
    records = assemble_coco_records(
        image_id=1,
        width=200,
        height=100,
        obj=obj,
        boxes=torch.tensor([[0.1, 0.1, 0.9, 0.9]]),
        quality=torch.tensor([0.5]),
        token_probabilities=torch.tensor([0.5]),
        pose_xy=pose,
        pose_visibility_logits=visibility,
        mask_logits=torch.zeros((0, 1, 1)),
    )
    assert records[0]["keypoints"][:3] == pytest.approx([100.0, 50.0, 0.0])


def test_person_id_repair_is_deterministic_and_auditable():
    obj = {
        "persons": [
            {
                "person_id": 2,
                "box": "<box>",
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 0,
                "age": 31,
            },
            {
                "person_id": 9,
                "box": "<box>",
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 1,
                "age": 20,
            },
        ]
    }
    repairs = canonicalize_person_ids_v1(obj)
    assert repairs == 2
    assert [person["person_id"] for person in obj["persons"]] == [1, 2]
    validate_prediction_object_v1(obj)


def test_inference_audit_reports_raw_schema_and_repairs_separately():
    summary = summarize_inference_audit(
        [
            {
                "schema_ok": True,
                "raw_schema_ok": True,
                "person_id_repairs": 0,
                "eos_seen": True,
                "health": {"ok": True},
            },
            {
                "schema_ok": True,
                "raw_schema_ok": False,
                "person_id_repairs": 2,
                "eos_seen": True,
                "health": {"ok": True},
            },
            {
                "schema_ok": False,
                "raw_schema_ok": False,
                "person_id_repairs": 0,
                "eos_seen": False,
                "health": {"ok": False},
            },
        ]
    )
    assert summary["schema_ok_rate"] == pytest.approx(2 / 3)
    assert summary["raw_schema_ok_rate"] == pytest.approx(1 / 3)
    assert summary["person_id_repair_image_rate"] == pytest.approx(1 / 3)
    assert summary["person_id_repairs"] == 2
    assert summary["health_ok_rate"] == pytest.approx(2 / 3)
    assert summary["eos_rate"] == pytest.approx(2 / 3)


def test_checkpoint_pose_decoder_type_is_versioned_and_backward_compatible(
    tmp_path,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    task_heads = checkpoint / "stage1_task_heads.pt"
    torch.save({"pose_decoder_type": "direct_joint_query"}, task_heads)
    assert (
        _pose_decoder_type_from_checkpoint(checkpoint)
        == "direct_joint_query"
    )
    torch.save({"pose_decoder_type": "topdown_heatmap"}, task_heads)
    assert (
        _pose_decoder_type_from_checkpoint(checkpoint)
        == "topdown_heatmap"
    )
    torch.save({}, task_heads)
    assert _pose_decoder_type_from_checkpoint(checkpoint) == "coordinate"
