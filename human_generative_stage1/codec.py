from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError


LOCATION_BINS = 1000
CONTOUR_POINT_BUDGET = 64
COCO_KEYPOINT_NAMES = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)


class CodecError(ValueError):
    pass


@dataclass(frozen=True)
class DecodedPerson:
    box: np.ndarray
    pose: np.ndarray
    contours: Tuple[np.ndarray, ...]
    gender: int
    age: int
    consumed_tokens: int


def loc_token(index: int) -> str:
    value = int(index)
    if not 0 <= value < LOCATION_BINS:
        raise CodecError(f"location index outside [0,999]: {value}")
    return f"<loc_{value:03d}>"


def parse_loc_token(token: str) -> int:
    if (
        not isinstance(token, str)
        or len(token) != 9
        or not token.startswith("<loc_")
        or not token.endswith(">")
    ):
        raise CodecError(f"invalid location token: {token!r}")
    payload = token[5:8]
    if not payload.isdigit():
        raise CodecError(f"invalid location token: {token!r}")
    value = int(payload)
    if not 0 <= value < LOCATION_BINS:
        raise CodecError(f"location token outside range: {token!r}")
    return value


def quantize_coordinate(value: float, extent: int) -> int:
    if int(extent) <= 0:
        raise CodecError(f"invalid coordinate extent: {extent}")
    normalized = float(value) / float(extent)
    return int(np.rint(np.clip(normalized, 0.0, 1.0) * (LOCATION_BINS - 1)))


def dequantize_coordinate(index: int, extent: int) -> float:
    if int(extent) <= 0:
        raise CodecError(f"invalid coordinate extent: {extent}")
    return float(int(index)) * float(extent) / float(LOCATION_BINS - 1)


def _as_xy_points(polygon: Sequence[float]) -> np.ndarray:
    values = np.asarray(polygon, dtype=np.float64)
    if values.ndim != 1 or values.size < 6 or values.size % 2:
        raise CodecError("polygon must contain at least three x-y pairs")
    points = values.reshape(-1, 2)
    finite = np.isfinite(points).all(axis=1)
    points = points[finite]
    if len(points) < 3:
        raise CodecError("polygon has fewer than three finite points")
    if np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise CodecError("polygon degenerates after closure removal")
    return points


def _closed_perimeter(points: np.ndarray) -> float:
    closed = np.concatenate([points, points[:1]], axis=0)
    return float(np.linalg.norm(np.diff(closed, axis=0), axis=1).sum())


def resample_closed_contour(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = int(count)
    if len(points) < 3 or count < 3:
        raise CodecError("closed contour resampling requires at least three points")
    closed = np.concatenate([points, points[:1]], axis=0)
    vectors = np.diff(closed, axis=0)
    lengths = np.linalg.norm(vectors, axis=1)
    keep = lengths > 1e-8
    if int(keep.sum()) < 3:
        raise CodecError("contour has fewer than three nonzero edges")
    starts = closed[:-1][keep]
    vectors = vectors[keep]
    lengths = lengths[keep]
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    total = float(cumulative[-1])
    samples = np.linspace(0.0, total, count, endpoint=False)
    edge_indices = np.searchsorted(cumulative[1:], samples, side="right")
    local = (samples - cumulative[edge_indices]) / lengths[edge_indices]
    return starts[edge_indices] + vectors[edge_indices] * local[:, None]


def allocate_contour_budget(
    contours: Sequence[np.ndarray],
    total_budget: int = CONTOUR_POINT_BUDGET,
) -> List[int]:
    total_budget = int(total_budget)
    valid = [np.asarray(value, dtype=np.float64) for value in contours]
    if not valid:
        return []
    max_contours = total_budget // 3
    if max_contours <= 0:
        raise CodecError("point budget cannot represent a contour")
    valid = valid[:max_contours]
    base = np.full(len(valid), 3, dtype=np.int64)
    remaining = total_budget - int(base.sum())
    if remaining <= 0:
        return base.tolist()
    perimeters = np.asarray(
        [max(_closed_perimeter(value), 1e-6) for value in valid],
        dtype=np.float64,
    )
    ideal = remaining * perimeters / perimeters.sum()
    extra = np.floor(ideal).astype(np.int64)
    remainder = remaining - int(extra.sum())
    if remainder:
        order = sorted(
            range(len(valid)),
            key=lambda index: (
                -(ideal[index] - extra[index]),
                -perimeters[index],
                index,
            ),
        )
        for index in order[:remainder]:
            extra[index] += 1
    counts = base + extra
    if int(counts.sum()) != total_budget:
        raise AssertionError("contour allocation does not conserve point budget")
    return counts.tolist()


def _mask_to_convex_component_contours(mask: np.ndarray) -> List[np.ndarray]:
    """Fallback for RLE masks when owner polygons are unavailable."""
    labeled, components = ndimage.label(np.asarray(mask, dtype=bool))
    contours: List[Tuple[float, int, int, np.ndarray]] = []
    for component_id in range(1, int(components) + 1):
        ys, xs = np.nonzero(labeled == component_id)
        if len(xs) < 3:
            continue
        points = np.stack([xs.astype(np.float64), ys.astype(np.float64)], axis=1)
        try:
            hull = ConvexHull(points)
            contour = points[hull.vertices]
        except QhullError:
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            contour = np.asarray(
                [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                dtype=np.float64,
            )
        area = float(len(xs))
        contours.append(
            (area, int(xs.min()), int(ys.min()), contour)
        )
    contours.sort(key=lambda value: (-value[0], value[1], value[2]))
    return [value[-1] for value in contours]


def contours_from_segmentation(
    segmentation: Any,
    *,
    height: int,
    width: int,
) -> List[np.ndarray]:
    contours: List[Tuple[float, float, float, np.ndarray]] = []
    if isinstance(segmentation, list):
        for polygon in segmentation:
            try:
                points = _as_xy_points(polygon)
            except CodecError:
                continue
            perimeter = _closed_perimeter(points)
            if perimeter <= 1e-6:
                continue
            x_min = float(points[:, 0].min())
            y_min = float(points[:, 1].min())
            contours.append((perimeter, x_min, y_min, points))
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, int(height), int(width))
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)
        for points in _mask_to_convex_component_contours(decoded):
            contours.append(
                (
                    _closed_perimeter(points),
                    float(points[:, 0].min()),
                    float(points[:, 1].min()),
                    points,
                )
            )
    elif segmentation:
        raise CodecError(f"unsupported segmentation type: {type(segmentation)}")
    contours.sort(key=lambda value: (-value[0], value[1], value[2]))
    return [value[-1] for value in contours]


def serialize_box(box_xyxy: Sequence[float], *, width: int, height: int) -> List[str]:
    if len(box_xyxy) != 4:
        raise CodecError("box must contain four coordinates")
    x1, y1, x2, y2 = [float(value) for value in box_xyxy]
    if x2 <= x1 or y2 <= y1:
        raise CodecError(f"invalid xyxy box: {box_xyxy}")
    return [
        "<box>",
        loc_token(quantize_coordinate(x1, width)),
        loc_token(quantize_coordinate(y1, height)),
        loc_token(quantize_coordinate(x2, width)),
        loc_token(quantize_coordinate(y2, height)),
    ]


def serialize_pose(
    keypoints: np.ndarray,
    *,
    width: int,
    height: int,
) -> List[str]:
    values = np.asarray(keypoints, dtype=np.float64)
    if values.shape != (17, 3):
        raise CodecError(f"expected COCO-17 keypoints, got {values.shape}")
    tokens = ["<pose>"]
    for x, y, visibility in values:
        vis = int(np.clip(np.rint(visibility), 0, 2))
        if vis == 0:
            x_token = loc_token(0)
            y_token = loc_token(0)
        else:
            x_token = loc_token(quantize_coordinate(float(x), width))
            y_token = loc_token(quantize_coordinate(float(y), height))
        tokens.extend([x_token, y_token, f"<vis_{vis}>"])
    return tokens


def serialize_contours(
    contours: Sequence[np.ndarray],
    *,
    width: int,
    height: int,
    point_budget: int = CONTOUR_POINT_BUDGET,
) -> List[str]:
    valid = [
        np.asarray(value, dtype=np.float64).reshape(-1, 2)
        for value in contours
        if np.asarray(value).size >= 6
    ]
    max_contours = int(point_budget) // 3
    valid = valid[:max_contours]
    tokens = ["<segment>"]
    if not valid:
        tokens.append("<empty_segment>")
        return tokens
    counts = allocate_contour_budget(valid, point_budget)
    for contour, count in zip(valid, counts):
        sampled = resample_closed_contour(contour, count)
        tokens.append("<contour>")
        for x, y in sampled:
            tokens.append(loc_token(quantize_coordinate(float(x), width)))
            tokens.append(loc_token(quantize_coordinate(float(y), height)))
        tokens.append("</contour>")
    return tokens


def gender_token(value: int, valid: bool) -> str:
    if not valid or int(value) not in (0, 1):
        return "<gender_unknown>"
    return "<gender_female>" if int(value) == 1 else "<gender_male>"


def age_token(value: float, valid: bool) -> str:
    if not valid or not np.isfinite(float(value)):
        return "<age_unknown>"
    age = int(np.clip(np.rint(float(value)), 0, 120))
    return f"<age_{age:03d}>"


def serialize_person(
    instance: Mapping[str, Any],
    *,
    width: int,
    height: int,
    segmentation: Any | None = None,
) -> List[str]:
    if segmentation is None:
        segmentation = instance.get("segmentation")
    contours = contours_from_segmentation(
        segmentation,
        height=height,
        width=width,
    )
    tokens = ["<person>"]
    tokens.extend(serialize_box(instance["bbox_xyxy"], width=width, height=height))
    tokens.extend(serialize_pose(instance["keypoints"], width=width, height=height))
    tokens.extend(
        serialize_contours(contours, width=width, height=height)
    )
    tokens.extend(
        [
            "<gender>",
            gender_token(
                int(instance.get("gender", -1)),
                bool(instance.get("gender_valid", False)),
            ),
            "<age>",
            age_token(
                float(instance.get("age", -1)),
                bool(instance.get("age_valid", False)),
            ),
            "</person>",
        ]
    )
    return tokens


def _expect(tokens: Sequence[str], cursor: int, expected: str) -> int:
    if cursor >= len(tokens) or tokens[cursor] != expected:
        actual = tokens[cursor] if cursor < len(tokens) else "<eos>"
        raise CodecError(
            f"expected {expected!r} at token {cursor}, found {actual!r}"
        )
    return cursor + 1


def _decode_gender(token: str) -> int:
    mapping = {
        "<gender_male>": 0,
        "<gender_female>": 1,
        "<gender_unknown>": -1,
    }
    if token not in mapping:
        raise CodecError(f"invalid gender token: {token}")
    return mapping[token]


def _decode_age(token: str) -> int:
    if token == "<age_unknown>":
        return -1
    if (
        len(token) != 9
        or not token.startswith("<age_")
        or not token.endswith(">")
        or not token[5:8].isdigit()
    ):
        raise CodecError(f"invalid age token: {token}")
    age = int(token[5:8])
    if not 0 <= age <= 120:
        raise CodecError(f"age token outside range: {token}")
    return age


def parse_person(
    tokens: Sequence[str],
    *,
    width: int,
    height: int,
    start: int = 0,
) -> DecodedPerson:
    cursor = _expect(tokens, int(start), "<person>")
    cursor = _expect(tokens, cursor, "<box>")
    box_indices = [parse_loc_token(tokens[cursor + offset]) for offset in range(4)]
    cursor += 4
    box = np.asarray(
        [
            dequantize_coordinate(box_indices[0], width),
            dequantize_coordinate(box_indices[1], height),
            dequantize_coordinate(box_indices[2], width),
            dequantize_coordinate(box_indices[3], height),
        ],
        dtype=np.float32,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        raise CodecError(f"decoded invalid box: {box.tolist()}")

    cursor = _expect(tokens, cursor, "<pose>")
    pose = np.zeros((17, 3), dtype=np.float32)
    for joint in range(17):
        x_index = parse_loc_token(tokens[cursor])
        y_index = parse_loc_token(tokens[cursor + 1])
        vis_token = tokens[cursor + 2]
        if vis_token not in {"<vis_0>", "<vis_1>", "<vis_2>"}:
            raise CodecError(f"invalid visibility token: {vis_token}")
        visibility = int(vis_token[5])
        if visibility:
            pose[joint, 0] = dequantize_coordinate(x_index, width)
            pose[joint, 1] = dequantize_coordinate(y_index, height)
        pose[joint, 2] = visibility
        cursor += 3

    cursor = _expect(tokens, cursor, "<segment>")
    contours: List[np.ndarray] = []
    if cursor < len(tokens) and tokens[cursor] == "<empty_segment>":
        cursor += 1
    else:
        while cursor < len(tokens) and tokens[cursor] == "<contour>":
            cursor += 1
            points = []
            while cursor < len(tokens) and tokens[cursor] != "</contour>":
                if cursor + 1 >= len(tokens):
                    raise CodecError("unterminated x-y contour pair")
                x_index = parse_loc_token(tokens[cursor])
                y_index = parse_loc_token(tokens[cursor + 1])
                points.append(
                    [
                        dequantize_coordinate(x_index, width),
                        dequantize_coordinate(y_index, height),
                    ]
                )
                cursor += 2
            cursor = _expect(tokens, cursor, "</contour>")
            if len(points) < 3:
                raise CodecError("decoded contour has fewer than three points")
            contours.append(np.asarray(points, dtype=np.float32))

    cursor = _expect(tokens, cursor, "<gender>")
    if cursor >= len(tokens):
        raise CodecError("missing gender value")
    gender = _decode_gender(tokens[cursor])
    cursor += 1
    cursor = _expect(tokens, cursor, "<age>")
    if cursor >= len(tokens):
        raise CodecError("missing age value")
    age = _decode_age(tokens[cursor])
    cursor += 1
    cursor = _expect(tokens, cursor, "</person>")
    return DecodedPerson(
        box=box,
        pose=pose,
        contours=tuple(contours),
        gender=gender,
        age=age,
        consumed_tokens=cursor - int(start),
    )


def serialize_people(
    people: Sequence[Tuple[Mapping[str, Any], Any]],
    *,
    width: int,
    height: int,
) -> List[str]:
    tokens: List[str] = []
    for instance, segmentation in people:
        tokens.extend(
            serialize_person(
                instance,
                width=width,
                height=height,
                segmentation=segmentation,
            )
        )
    return tokens


def parse_people(
    tokens: Sequence[str],
    *,
    width: int,
    height: int,
) -> List[DecodedPerson]:
    people = []
    cursor = 0
    while cursor < len(tokens):
        person = parse_person(
            tokens,
            width=width,
            height=height,
            start=cursor,
        )
        people.append(person)
        cursor += person.consumed_tokens
    return people


def rasterize_contours(
    contours: Sequence[np.ndarray],
    *,
    height: int,
    width: int,
) -> np.ndarray:
    canvas = Image.new("1", (int(width), int(height)), 0)
    draw = ImageDraw.Draw(canvas)
    for contour in contours:
        points = np.asarray(contour, dtype=np.float64).reshape(-1, 2)
        if len(points) < 3:
            continue
        polygon = [
            (
                int(np.rint(np.clip(x, 0, width - 1))),
                int(np.rint(np.clip(y, 0, height - 1))),
            )
            for x, y in points
        ]
        draw.polygon(polygon, fill=1)
    return np.asarray(canvas, dtype=bool)


def segmentation_to_mask(
    segmentation: Any,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if not segmentation:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, list):
        rles = mask_utils.frPyObjects(segmentation, int(height), int(width))
        rle = mask_utils.merge(rles)
    elif isinstance(segmentation, dict):
        rle = segmentation
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, int(height), int(width))
    else:
        raise CodecError(f"unsupported segmentation type: {type(segmentation)}")
    decoded = mask_utils.decode(rle)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return np.asarray(decoded, dtype=bool)


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=bool)
    second = np.asarray(second, dtype=bool)
    intersection = np.logical_and(first, second).sum()
    union = np.logical_or(first, second).sum()
    return float(intersection / union) if union else 1.0
