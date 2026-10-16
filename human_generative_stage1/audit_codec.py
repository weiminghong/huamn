#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from human_generative_stage1.codec import (
    contours_from_segmentation,
    mask_iou,
    parse_person,
    rasterize_contours,
    segmentation_to_mask,
    serialize_person,
)


def as_bool(value: Any) -> bool:
    try:
        return bool(int(value))
    except (TypeError, ValueError):
        return bool(value)


def valid_box(annotation: Dict[str, Any]) -> bool:
    box = annotation.get("bbox")
    if not isinstance(box, list) or len(box) != 4:
        return False
    x, y, width, height = [float(value) for value in box]
    return (
        np.isfinite([x, y, width, height]).all()
        and width > 0
        and height > 0
    )


def to_instance(annotation: Dict[str, Any]) -> Dict[str, Any]:
    x, y, width, height = [float(value) for value in annotation["bbox"]]
    keypoints = np.zeros((17, 3), dtype=np.float32)
    values = annotation.get("keypoints")
    if isinstance(values, list) and len(values) >= 51:
        candidate = np.asarray(values[:51], dtype=np.float32).reshape(17, 3)
        if np.isfinite(candidate).all():
            keypoints = candidate
    valid = annotation.get("valid") or {}
    gender_valid = as_bool(valid.get("gender", 0))
    age_valid = as_bool(valid.get("age", 0))
    return {
        "bbox_xyxy": [x, y, x + width, y + height],
        "keypoints": keypoints,
        "gender": int(round(float(annotation.get("gender", -1))))
        if gender_valid
        else -1,
        "gender_valid": gender_valid,
        "age": float(annotation.get("age", -1)) if age_valid else -1.0,
        "age_valid": age_valid,
    }


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "min": None,
            "p05": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p05": float(np.percentile(array, 5)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(array.max()),
    }


def mask_digest(mask: np.ndarray) -> str:
    return hashlib.sha256(np.packbits(np.asarray(mask, dtype=np.uint8)).tobytes()).hexdigest()


def estimated_person_tokens(annotation: Dict[str, Any]) -> int:
    segmentation = annotation.get("segmentation")
    if isinstance(segmentation, list):
        contour_count = min(
            21,
            sum(
                1
                for polygon in segmentation
                if isinstance(polygon, list)
                and len(polygon) >= 6
                and len(polygon) % 2 == 0
            ),
        )
    elif isinstance(segmentation, dict):
        contour_count = 1
    else:
        contour_count = 0
    if contour_count == 0:
        return 65
    return 192 + 2 * contour_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotation",
        default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json",
    )
    parser.add_argument("--polygon-samples", type=int, default=256)
    parser.add_argument("--rle-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.annotation).read_text(encoding="utf-8"))
    images = {int(item["id"]): item for item in payload["images"]}
    eligible = []
    token_lengths = defaultdict(int)
    person_counts = defaultdict(int)

    for annotation in payload["annotations"]:
        if (
            as_bool(annotation.get("iscrowd", 0))
            or as_bool(annotation.get("ignore", 0))
            or not valid_box(annotation)
        ):
            continue
        segmentation = annotation.get("segmentation")
        if segmentation:
            eligible.append(annotation)
        image_id = int(annotation["image_id"])
        person_counts[image_id] += 1
        token_lengths[image_id] += estimated_person_tokens(annotation)

    polygon = [
        item for item in eligible if isinstance(item.get("segmentation"), list)
    ]
    rle = [
        item for item in eligible if isinstance(item.get("segmentation"), dict)
    ]
    randomizer = random.Random(args.seed)
    selected = []
    selected.extend(
        ("polygon", item)
        for item in randomizer.sample(
            polygon, min(args.polygon_samples, len(polygon))
        )
    )
    selected.extend(
        ("rle", item)
        for item in randomizer.sample(rle, min(args.rle_samples, len(rle)))
    )

    ious: Dict[str, List[float]] = {"polygon": [], "rle": [], "all": []}
    token_counts = []
    box_errors = []
    pose_errors = []
    deterministic = True
    nonempty = True
    reconstructed_digests = []
    low_iou = []

    for kind, annotation in selected:
        image = images[int(annotation["image_id"])]
        width = int(image["width"])
        height = int(image["height"])
        instance = to_instance(annotation)
        segmentation = annotation["segmentation"]
        first = serialize_person(
            instance,
            width=width,
            height=height,
            segmentation=segmentation,
        )
        second = serialize_person(
            instance,
            width=width,
            height=height,
            segmentation=segmentation,
        )
        deterministic = deterministic and first == second
        decoded = parse_person(first, width=width, height=height)
        target_mask = segmentation_to_mask(
            segmentation,
            height=height,
            width=width,
        )
        reconstructed = rasterize_contours(
            decoded.contours,
            height=height,
            width=width,
        )
        nonempty = nonempty and bool(target_mask.any()) and bool(reconstructed.any())
        current_iou = mask_iou(target_mask, reconstructed)
        ious[kind].append(current_iou)
        ious["all"].append(current_iou)
        token_counts.append(len(first))
        reconstructed_digests.append(mask_digest(reconstructed))

        target_box = np.asarray(instance["bbox_xyxy"], dtype=np.float32)
        box_errors.append(float(np.abs(decoded.box - target_box).max()))
        visible = instance["keypoints"][:, 2] > 0
        if visible.any():
            pose_errors.append(
                float(
                    np.abs(
                        decoded.pose[visible, :2]
                        - instance["keypoints"][visible, :2]
                    ).max()
                )
            )
        if current_iou < 0.7:
            low_iou.append(
                {
                    "kind": kind,
                    "annotation_id": int(annotation["id"]),
                    "image_id": int(annotation["image_id"]),
                    "iou": current_iou,
                    "tokens": len(first),
                    "source_contours": len(
                        contours_from_segmentation(
                            segmentation,
                            height=height,
                            width=width,
                        )
                    ),
                }
            )

    all_lengths = list(token_lengths.values())
    all_person_counts = list(person_counts.values())
    unique_ratio = (
        len(set(reconstructed_digests)) / len(reconstructed_digests)
        if reconstructed_digests
        else 0.0
    )

    report = {
        "schema": "human-generative-codec-audit-v1",
        "annotation": args.annotation,
        "seed": args.seed,
        "eligible_instances": {
            "polygon": len(polygon),
            "rle": len(rle),
        },
        "sampled_instances": {
            "polygon": len(ious["polygon"]),
            "rle": len(ious["rle"]),
            "all": len(ious["all"]),
        },
        "mask_iou": {
            key: summarize(value) for key, value in ious.items()
        },
        "instance_token_count": summarize(
            [float(value) for value in token_counts]
        ),
        "full_image_estimated_token_count": summarize(
            [float(value) for value in all_lengths]
        ),
        "full_image_person_count": summarize(
            [float(value) for value in all_person_counts]
        ),
        "images_over_5000_estimated_tokens": sum(
            value > 5000 for value in all_lengths
        ),
        "max_box_quantization_error_pixels": max(box_errors, default=0.0),
        "max_pose_quantization_error_pixels": max(pose_errors, default=0.0),
        "deterministic": deterministic,
        "nonempty": nonempty,
        "distinct_reconstruction_ratio": unique_ratio,
        "low_iou_examples": sorted(
            low_iou,
            key=lambda value: value["iou"],
        )[:20],
    }

    criteria = {
        "deterministic": deterministic,
        "all_reconstructions_nonempty": nonempty,
        "distinct_reconstruction_ratio_ge_0_99": unique_ratio >= 0.99,
        "polygon_mean_iou_ge_0_85": report["mask_iou"]["polygon"]["mean"] >= 0.85,
        "rle_mean_iou_ge_0_70_or_not_applicable": (
            report["mask_iou"]["rle"]["count"] == 0
            or report["mask_iou"]["rle"]["mean"] >= 0.70
        ),
        "overall_mean_iou_ge_0_80": report["mask_iou"]["all"]["mean"] >= 0.80,
        "box_quantization_error_le_1px": report[
            "max_box_quantization_error_pixels"
        ] <= 1.0,
        "pose_quantization_error_le_1px": report[
            "max_pose_quantization_error_pixels"
        ] <= 1.0,
    }
    report["criteria"] = criteria
    report["pass"] = all(criteria.values())

    output = json.dumps(report, indent=2) + "\n"
    Path(args.output).write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
