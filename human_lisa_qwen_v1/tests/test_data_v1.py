import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.data_v1 import (
    COCO_FLIP_INDEX,
    CocoUniHumanV1Dataset,
    audit_annotation_file,
    build_instances,
    horizontal_flip_instances,
)


def _valid(**updates):
    value = {
        "labels": 1,
        "bboxes": 1,
        "keypoints_2d": 1,
        "segmentation": 1,
        "smpl": 0,
        "gender": 0,
        "age": 0,
        "face": 0,
    }
    value.update(updates)
    return value


def _keypoints(x_offset):
    out = []
    for index in range(17):
        out.extend([float(x_offset + index), float(10 + index), 2])
    return out


def _ann(ann_id, bbox, *, crowd=0, gender=None, age=None, valid=None):
    item = {
        "id": ann_id,
        "image_id": 7,
        "category_id": 1,
        "bbox": list(bbox),
        "area": float(bbox[2] * bbox[3]),
        "iscrowd": crowd,
        "keypoints": _keypoints(int(bbox[0])),
        "num_keypoints": 17,
        "segmentation": [
            [
                bbox[0],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1],
                bbox[0] + bbox[2],
                bbox[1] + bbox[3],
                bbox[0],
                bbox[1] + bbox[3],
            ]
        ],
        "valid": valid or _valid(),
    }
    if gender is not None:
        item["gender"] = gender
        item["valid"]["gender"] = 1
    if age is not None:
        item["age"] = age
        item["valid"]["age"] = 1
    return item


def test_build_instances_filters_and_binds_after_sorting():
    annotations = [
        _ann(30, [70, 5, 20, 30], gender=1),
        _ann(10, [10, 10, 20, 30], gender=0, age=21),
        _ann(20, [40, 10, 20, 30], crowd=1),
        _ann(40, [20, 15, 5, 0]),
    ]
    instances, skipped = build_instances(
        annotations, width=100, height=80, include_eval_segmentation=False
    )

    assert [item["source_ann_id"] for item in instances] == [10, 30]
    assert [item["person_id"] for item in instances] == [1, 2]
    assert instances[0]["bbox_xyxy"] == [10.0, 10.0, 30.0, 40.0]
    assert instances[0]["gender"] == 0
    assert instances[0]["gender_valid"] is True
    assert instances[0]["age"] == 21.0
    assert instances[0]["age_valid"] is True
    assert instances[1]["age_valid"] is False
    assert instances[0]["pose_valid"] is True
    assert instances[0]["seg_valid"] is True
    assert instances[0]["mask"].shape == (80, 100)
    assert instances[0]["mask"].dtype == np.bool_
    assert skipped == {"crowd": 1, "invalid_box": 1}


def test_eval_can_decode_present_polygons_when_valid_flag_is_zero():
    ann = _ann(
        11,
        [10, 10, 20, 30],
        valid=_valid(segmentation=0),
    )
    train_instances, _ = build_instances(
        [ann], width=100, height=80, include_eval_segmentation=False
    )
    eval_instances, _ = build_instances(
        [ann], width=100, height=80, include_eval_segmentation=True
    )
    assert train_instances[0]["seg_valid"] is False
    assert train_instances[0]["mask"] is None
    assert eval_instances[0]["seg_valid"] is True
    assert eval_instances[0]["mask"].any()


def test_build_instances_omits_pose_without_visible_keypoints():
    ann = _ann(12, [10, 10, 20, 30])
    ann["num_keypoints"] = 0
    ann["keypoints"] = [0.0] * 51
    instances, skipped = build_instances(
        [ann], width=100, height=80, include_eval_segmentation=False
    )

    assert skipped == {}
    assert instances[0]["pose_valid"] is False
    np.testing.assert_array_equal(
        instances[0]["keypoints"], np.zeros((17, 3), dtype=np.float32)
    )


def test_horizontal_flip_keeps_box_pose_mask_and_order_aligned():
    instances, _ = build_instances(
        [
            _ann(1, [10, 10, 20, 30]),
            _ann(2, [70, 10, 20, 30]),
        ],
        width=100,
        height=80,
        include_eval_segmentation=False,
    )
    original_first_pose = instances[0]["keypoints"].copy()
    flipped = horizontal_flip_instances(instances, width=100)

    assert [item["source_ann_id"] for item in flipped] == [2, 1]
    assert [item["person_id"] for item in flipped] == [1, 2]
    assert flipped[0]["bbox_xyxy"] == [10.0, 10.0, 30.0, 40.0]
    expected_x = 100.0 - original_first_pose[COCO_FLIP_INDEX, 0]
    np.testing.assert_allclose(flipped[1]["keypoints"][:, 0], expected_x)
    np.testing.assert_array_equal(
        flipped[1]["mask"], np.fliplr(instances[0]["mask"])
    )


def test_dataset_reads_official_style_json(tmp_path):
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    from PIL import Image

    Image.new("RGB", (100, 80), color="white").save(image_dir / "x.jpg")
    payload = {
        "images": [{"id": 7, "file_name": "x.jpg", "width": 100, "height": 80}],
        "annotations": [_ann(1, [10, 10, 20, 30])],
        "categories": [{"id": 1, "name": "person"}],
    }
    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(payload), encoding="utf-8")

    dataset = CocoUniHumanV1Dataset(
        str(ann_file), str(image_dir), training=True, horizontal_flip_prob=0.0
    )
    sample = dataset[0]
    assert sample["id"] == 7
    assert sample["image"].size == (100, 80)
    assert len(sample["instances"]) == 1
    assert sample["instances"][0]["source_ann_id"] == 1


@pytest.mark.slow
def test_official_v1_counts_and_non_crowd_maxima():
    train = audit_annotation_file(
        "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json"
    )
    val = audit_annotation_file(
        "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
    )
    assert (train["images"], train["annotations"]) == (64115, 262465)
    assert (val["images"], val["annotations"]) == (2693, 11004)
    assert train["max_noncrowd_per_image"] == 19
    assert val["max_noncrowd_per_image"] == 13
    assert train["invalid_box"] == 1
    assert train["valid"]["segmentation"] == 262465
    assert val["present_segmentation"] == 11004
