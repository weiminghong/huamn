from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping

from .manifest import write_jsonl
from .schema import canonical_pose, empty_label


def as_bool(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def mask_label(annotation: Mapping) -> Dict:
    segmentation = annotation.get("segmentation")
    valid = annotation.get("valid") or {}
    if not segmentation or not as_bool(valid.get("segmentation", 0)):
        return empty_label()
    if isinstance(segmentation, list):
        encoding = {"type": "polygons", "data": segmentation}
    elif isinstance(segmentation, dict):
        encoding = {"type": "coco_rle", "data": segmentation}
    else:
        return empty_label()
    return {"valid": True, "encoding": encoding}


def pose_label(annotation: Mapping) -> Dict:
    valid = annotation.get("valid") or {}
    values = annotation.get("keypoints")
    if (
        not as_bool(valid.get("keypoints_2d", 0))
        or not isinstance(values, list)
        or len(values) < 51
        or int(annotation.get("num_keypoints", 0)) <= 0
    ):
        return empty_label()
    points = [values[index : index + 3] for index in range(0, 51, 3)]
    return canonical_pose(points, "coco17")


def scalar_label(annotation: Mapping, name: str) -> Dict:
    valid = annotation.get("valid") or {}
    value = finite(annotation.get(name))
    if not as_bool(valid.get(name, 0)) or value is None:
        return empty_label()
    if name == "gender":
        value = int(round(value))
        if value not in (0, 1):
            return empty_label()
    return {"valid": True, "value": value}


def convert(
    annotation: str,
    image_root: str,
    *,
    split: str,
    max_images: int = 0,
) -> Iterator[Dict]:
    with Path(annotation).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    grouped = defaultdict(list)
    for item in payload.get("annotations", []):
        grouped[int(item["image_id"])].append(item)
    images = payload.get("images", [])
    if max_images:
        images = images[: int(max_images)]
    for image in images:
        image_id = int(image["id"])
        width, height = int(image["width"]), int(image["height"])
        persons = []
        for annotation_item in grouped.get(image_id, []):
            if as_bool(annotation_item.get("ignore", 0)) or as_bool(
                annotation_item.get("iscrowd", 0)
            ):
                continue
            bbox = annotation_item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, box_width, box_height = map(float, bbox)
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(float(width), x + box_width), min(
                float(height), y + box_height
            )
            if x2 <= x1 or y2 <= y1:
                continue
            persons.append(
                {
                    "source_instance_id": str(annotation_item["id"]),
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "area": float(
                        annotation_item.get(
                            "area", (x2 - x1) * (y2 - y1)
                        )
                    ),
                    "labels": {
                        "pose": pose_label(annotation_item),
                        "segment": mask_label(annotation_item),
                        "parsing": empty_label(),
                        "age": scalar_label(annotation_item, "age"),
                        "gender": scalar_label(annotation_item, "gender"),
                        "attribute": empty_label(),
                    },
                    "provenance": {
                        "dataset": "coco_unihuman",
                        "annotation_id": int(annotation_item["id"]),
                    },
                }
            )
        persons.sort(
            key=lambda item: (
                (item["bbox_xyxy"][0] + item["bbox_xyxy"][2]) / 2,
                (item["bbox_xyxy"][1] + item["bbox_xyxy"][3]) / 2,
                int(item["source_instance_id"]),
            )
        )
        for slot_id, person in enumerate(persons, start=1):
            person["slot_id"] = slot_id
        yield {
            "schema": "human-multitask-v5",
            "uid": f"coco_unihuman:{split}:{image_id}",
            "dataset": "coco_unihuman",
            "split": split,
            "image": {
                "path": str(Path(image_root) / image["file_name"]),
                "width": width,
                "height": height,
                "source_image_id": image_id,
                "coco_id": image_id,
                "file_name": image["file_name"],
            },
            "persons": persons,
            "entities": [],
            "relations": [],
            "provenance": {
                "annotation_file": str(Path(annotation).resolve())
            },
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-images", type=int, default=0)
    args = parser.parse_args()
    result = write_jsonl(
        args.output,
        convert(
            args.annotation,
            args.image_root,
            split=args.split,
            max_images=args.max_images,
        ),
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
