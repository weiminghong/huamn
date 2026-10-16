import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.official_eval_v1 import (
    encode_binary_mask,
    evaluate_all_official,
)


def _toy_annotations(path):
    keypoint_names = [f"kpt_{index}" for index in range(17)]
    keypoints = []
    for index in range(17):
        keypoints.extend([20 + index, 20 + index, 2])
    payload = {
        "images": [
            {"id": 1, "file_name": "one.jpg", "width": 100, "height": 100}
        ],
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": keypoint_names,
                "skeleton": [],
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 60, 70],
                "area": 4200,
                "iscrowd": 0,
                "segmentation": [[10, 10, 70, 10, 70, 80, 10, 80]],
                "keypoints": keypoints,
                "num_keypoints": 17,
                "gender": 0,
                "age": 25,
                "valid": {
                    "labels": 1,
                    "bboxes": 1,
                    "keypoints_2d": 1,
                    "segmentation": 1,
                    "gender": 1,
                    "age": 1,
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return keypoints


def test_binary_mask_encoding_round_trip():
    from pycocotools import mask as mask_utils

    mask = np.zeros((20, 30), dtype=np.bool_)
    mask[3:15, 5:25] = True
    encoded = encode_binary_mask(mask)
    assert isinstance(encoded["counts"], str)
    decoded = mask_utils.decode(
        {"size": encoded["size"], "counts": encoded["counts"].encode("ascii")}
    )
    np.testing.assert_array_equal(decoded.astype(bool), mask)


def test_perfect_predictions_score_one_under_official_evaluators(tmp_path):
    ann_file = tmp_path / "toy.json"
    keypoints = _toy_annotations(ann_file)
    mask = np.zeros((100, 100), dtype=np.bool_)
    mask[10:80, 10:70] = True
    predictions = [
        {
            "image_id": 1,
            "category_id": 1,
            "bbox": [10.0, 10.0, 60.0, 70.0],
            "score": 0.99,
            "keypoints": [float(value) for value in keypoints],
            "segmentation": encode_binary_mask(mask),
            "gender": 0.0,
            "age": 25.0,
        }
    ]
    metrics = evaluate_all_official(
        str(ann_file),
        predictions,
        image_ids=[1],
        attr_eval_source=(
            REPO_ROOT
            / "HumanQueryNet_mmdet3_port"
            / "datasets"
            / "api_wrappers"
            / "eval_attr.py"
        ),
    )
    assert metrics["bbox_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["bbox_AP50"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["keypoints_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["keypoints_AP50"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["segm_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["segm_AP50"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["gender_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["age_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["gt_persons"] == 1
    assert metrics["gender_gt_labeled"] == 1
    assert metrics["age_gt_labeled"] == 1
    assert metrics["gender_gt_coverage"] == pytest.approx(1.0)
    assert metrics["age_gt_coverage"] == pytest.approx(1.0)


def test_empty_predictions_return_zero_not_crash(tmp_path):
    ann_file = tmp_path / "toy.json"
    _toy_annotations(ann_file)
    metrics = evaluate_all_official(
        str(ann_file),
        [],
        image_ids=[1],
        attr_eval_source=(
            REPO_ROOT
            / "HumanQueryNet_mmdet3_port"
            / "datasets"
            / "api_wrappers"
            / "eval_attr.py"
        ),
    )
    for name in [
        "bbox_AP",
        "keypoints_AP",
        "segm_AP",
        "gender_AP",
        "age_AP",
    ]:
        assert metrics[name] == 0.0



def test_attribute_eval_ignores_persons_without_attribute_labels(tmp_path):
    ann_file = tmp_path / "partial_attr.json"
    keypoints = _toy_annotations(ann_file)
    payload = json.loads(ann_file.read_text(encoding="utf-8"))
    payload["annotations"].append({
        "id": 2, "image_id": 1, "category_id": 1,
        "bbox": [72, 10, 20, 70], "area": 1400, "iscrowd": 0,
        "segmentation": [[72, 10, 92, 10, 92, 80, 72, 80]],
        "keypoints": keypoints, "num_keypoints": 17,
        "gender": -1, "age": -1,
        "valid": {"labels": 1, "bboxes": 1, "keypoints_2d": 1, "segmentation": 1, "gender": 0, "age": 0},
    })
    ann_file.write_text(json.dumps(payload), encoding="utf-8")
    predictions = [
        {"image_id": 1, "category_id": 1, "bbox": [10.0, 10.0, 60.0, 70.0], "score": 0.99, "gender": 0.0, "age": 25.0},
        {"image_id": 1, "category_id": 1, "bbox": [72.0, 10.0, 20.0, 70.0], "score": 0.98, "gender": 1.0, "age": 40.0},
    ]
    metrics = evaluate_all_official(
        str(ann_file), predictions, image_ids=[1],
        attr_eval_source=REPO_ROOT / "HumanQueryNet_mmdet3_port" / "datasets" / "api_wrappers" / "eval_attr.py",
    )
    assert metrics["gender_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["age_AP"] == pytest.approx(1.0, abs=1e-6)
    assert metrics["gt_persons"] == 2
    assert metrics["gender_gt_labeled"] == 1
    assert metrics["age_gt_labeled"] == 1
    assert metrics["gender_gt_coverage"] == pytest.approx(0.5)
    assert metrics["age_gt_coverage"] == pytest.approx(0.5)
