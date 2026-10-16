import numpy as np
import pytest
from pycocotools import mask as mask_utils

from human_generative_stage1.codec import (
    COCO_KEYPOINT_NAMES,
    CodecError,
    allocate_contour_budget,
    contours_from_segmentation,
    dequantize_coordinate,
    loc_token,
    mask_iou,
    parse_loc_token,
    parse_people,
    parse_person,
    quantize_coordinate,
    rasterize_contours,
    resample_closed_contour,
    segmentation_to_mask,
    serialize_contours,
    serialize_people,
    serialize_person,
)


def synthetic_instance(offset=0.0):
    pose = np.zeros((17, 3), dtype=np.float32)
    for index in range(17):
        pose[index] = [20.0 + index + offset, 30.0 + index, 2]
    return {
        "bbox_xyxy": [10.0 + offset, 15.0, 90.0 + offset, 110.0],
        "keypoints": pose,
        "gender": 1,
        "gender_valid": True,
        "age": 27.4,
        "age_valid": True,
    }


def square_polygon(offset=0.0):
    return [[
        10.0 + offset,
        15.0,
        90.0 + offset,
        15.0,
        90.0 + offset,
        110.0,
        10.0 + offset,
        110.0,
    ]]


def test_location_tokens_round_trip():
    assert loc_token(0) == "<loc_000>"
    assert loc_token(999) == "<loc_999>"
    for value in [0.0, 1.0, 99.5, 200.0]:
        index = quantize_coordinate(value, 200)
        decoded = dequantize_coordinate(index, 200)
        assert abs(decoded - np.clip(value, 0, 200)) <= 200 / 999 + 1e-6
    with pytest.raises(CodecError):
        parse_loc_token("<loc_1000>")


def test_contour_budget_and_resampling_are_deterministic():
    contours = [
        np.asarray([[0, 0], [20, 0], [20, 10], [0, 10]], dtype=float),
        np.asarray([[30, 30], [40, 30], [40, 40], [30, 40]], dtype=float),
    ]
    counts = allocate_contour_budget(contours, 64)
    assert sum(counts) == 64
    assert min(counts) >= 3
    first = resample_closed_contour(contours[0], counts[0])
    second = resample_closed_contour(contours[0], counts[0])
    np.testing.assert_array_equal(first, second)


def test_person_serialization_uses_native_token_sequence():
    instance = synthetic_instance()
    segmentation = square_polygon()
    tokens = serialize_person(
        instance,
        width=128,
        height=128,
        segmentation=segmentation,
    )
    assert tokens[0] == "<person>"
    assert tokens[-1] == "</person>"
    assert tokens.count("<box>") == 1
    assert tokens.count("<pose>") == 1
    assert tokens.count("<segment>") == 1
    assert tokens.count("<contour>") == 1
    assert sum(token.startswith("<loc_") for token in tokens) == 4 + 34 + 128

    decoded = parse_person(tokens, width=128, height=128)
    assert decoded.consumed_tokens == len(tokens)
    assert decoded.gender == 1
    assert decoded.age == 27
    assert decoded.pose.shape == (17, 3)
    assert tuple(decoded.pose[:, 2].astype(int)) == (2,) * 17
    predicted_mask = rasterize_contours(
        decoded.contours,
        height=128,
        width=128,
    )
    target_mask = segmentation_to_mask(
        segmentation,
        height=128,
        width=128,
    )
    assert mask_iou(predicted_mask, target_mask) > 0.95


def test_multiple_people_are_distinct_and_parse_exactly():
    first = synthetic_instance(0.0)
    second = synthetic_instance(20.0)
    tokens = serialize_people(
        [
            (first, square_polygon(0.0)),
            (second, square_polygon(20.0)),
        ],
        width=160,
        height=128,
    )
    decoded = parse_people(tokens, width=160, height=128)
    assert len(decoded) == 2
    assert decoded[0].box[0] < decoded[1].box[0]
    first_mask = rasterize_contours(decoded[0].contours, height=128, width=160)
    second_mask = rasterize_contours(decoded[1].contours, height=128, width=160)
    assert not np.array_equal(first_mask, second_mask)


def test_rle_fallback_is_deterministic_and_nonempty():
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[8:40, 10:30] = 1
    mask[42:58, 44:60] = 1
    encoded = mask_utils.encode(np.asfortranarray(mask))
    encoded["counts"] = encoded["counts"].decode("ascii")
    first = contours_from_segmentation(encoded, height=64, width=64)
    second = contours_from_segmentation(encoded, height=64, width=64)
    assert len(first) == 2
    assert len(second) == 2
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    tokens = serialize_contours(first, width=64, height=64)
    instance = synthetic_instance()
    full = serialize_person(
        instance,
        width=64,
        height=64,
        segmentation=encoded,
    )
    decoded = parse_person(full, width=64, height=64)
    reconstructed = rasterize_contours(
        decoded.contours,
        height=64,
        width=64,
    )
    assert reconstructed.any()
    assert "<segment>" in tokens


def test_joint_order_and_parser_fail_closed():
    assert len(COCO_KEYPOINT_NAMES) == 17
    tokens = serialize_person(
        synthetic_instance(),
        width=128,
        height=128,
        segmentation=square_polygon(),
    )
    corrupted = list(tokens)
    corrupted[corrupted.index("<pose>") + 3] = "<vis_9>"
    with pytest.raises(CodecError):
        parse_person(corrupted, width=128, height=128)
