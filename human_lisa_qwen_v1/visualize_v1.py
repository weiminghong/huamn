from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as mask_utils


COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

COLORS = (
    (220, 40, 40),
    (30, 120, 210),
    (30, 160, 80),
    (230, 150, 20),
    (150, 60, 190),
    (20, 170, 170),
)


def _decode_rle(segmentation: Dict[str, Any]) -> np.ndarray:
    encoded = dict(segmentation)
    if isinstance(encoded.get("counts"), str):
        encoded["counts"] = encoded["counts"].encode("ascii")
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = decoded[..., 0]
    return np.asarray(decoded, dtype=np.bool_)


def _draw_mask(
    canvas: Image.Image,
    segmentation: Dict[str, Any],
    color: tuple[int, int, int],
) -> None:
    mask = _decode_rle(segmentation)
    if mask.shape != (canvas.height, canvas.width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image "
            f"{(canvas.height, canvas.width)}"
        )
    alpha = Image.fromarray((mask.astype(np.uint8) * 80), mode="L")
    layer = Image.new("RGBA", canvas.size, (*color, 0))
    layer.putalpha(alpha)
    canvas.alpha_composite(layer)


def _draw_pose(
    draw: ImageDraw.ImageDraw,
    keypoints: Sequence[float],
    color: tuple[int, int, int],
) -> None:
    array = np.asarray(keypoints, dtype=np.float32).reshape(17, 3)
    visible = array[:, 2] > 0
    for first, second in COCO_SKELETON:
        if visible[first] and visible[second]:
            draw.line(
                [
                    (float(array[first, 0]), float(array[first, 1])),
                    (float(array[second, 0]), float(array[second, 1])),
                ],
                fill=(*color, 255),
                width=2,
            )
    radius = 2
    for x, y, visibility in array:
        if visibility <= 0:
            continue
        draw.ellipse(
            (float(x) - radius, float(y) - radius, float(x) + radius, float(y) + radius),
            fill=(*color, 255),
        )


def render_prediction_overlay(
    image: Image.Image,
    records: Sequence[Dict[str, Any]],
    output_path: Path | str,
    *,
    title: str = "",
) -> Dict[str, int]:
    canvas = image.convert("RGBA")
    mask_count = 0
    for index, record in enumerate(records):
        segmentation = record.get("segmentation")
        if isinstance(segmentation, dict):
            _draw_mask(canvas, segmentation, COLORS[index % len(COLORS)])
            mask_count += 1

    draw = ImageDraw.Draw(canvas)
    box_count = 0
    pose_count = 0
    for index, record in enumerate(records):
        color = COLORS[index % len(COLORS)]
        box = record.get("bbox")
        if isinstance(box, (list, tuple)) and len(box) == 4:
            x, y, width, height = [float(value) for value in box]
            draw.rectangle(
                (x, y, x + width, y + height),
                outline=(*color, 255),
                width=3,
            )
            label = (
                f"id={record.get('person_id', index + 1)} "
                f"s={float(record.get('score', 0.0)):.3f}"
            )
            draw.text((x + 2, max(0.0, y + 2)), label, fill=(*color, 255))
            box_count += 1
        keypoints = record.get("keypoints")
        if isinstance(keypoints, (list, tuple)) and len(keypoints) == 51:
            _draw_pose(draw, keypoints, color)
            pose_count += 1
    if title:
        draw.rectangle((0, 0, min(canvas.width, 420), 18), fill=(0, 0, 0, 170))
        draw.text((4, 3), title, fill=(255, 255, 255, 255))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(output)
    return {
        "persons": len(records),
        "boxes": box_count,
        "poses": pose_count,
        "masks": mask_count,
    }


def render_probe_overlays(
    samples: Sequence[Dict[str, Any]],
    results: Sequence[Dict[str, Any]],
    output_dir: Path | str,
    *,
    step: int,
) -> Dict[str, Any]:
    if len(samples) != len(results):
        raise ValueError("probe samples and results must have equal lengths")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    person_count = 0
    for sample, result in zip(samples, results):
        image_id = int(result["image_id"])
        if int(sample["id"]) != image_id:
            raise ValueError("probe sample/result image ids do not match")
        records = result.get("records", [])
        file_name = f"step-{int(step)}-image-{image_id}.png"
        health_ok = bool(result.get("health", {}).get("ok", False))
        schema_ok = bool(result.get("schema_ok", False))
        render_prediction_overlay(
            sample["image"],
            records,
            output / file_name,
            title=(
                f"step={int(step)} image={image_id} "
                f"schema={int(schema_ok)} health={int(health_ok)}"
            ),
        )
        files.append(file_name)
        person_count += len(records)
    return {
        "images": len(results),
        "persons": person_count,
        "files": files,
    }
