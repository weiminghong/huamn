from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from human_lisa_qwen_v4.collator_v4 import Stage1V4Collator


class Stage1TransferCollator(Stage1V4Collator):
    """Keep the v4 text contract while exposing person-aligned dense targets."""

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch = super().__call__(samples)
        dense_targets: List[Dict[str, Any]] = []
        for sample in samples:
            width = float(sample["width"])
            height = float(sample["height"])
            instances = sample["instances"]
            boxes = []
            poses = []
            pose_valid = []
            masks = []
            seg_valid = []
            genders = []
            gender_valid = []
            ages = []
            age_valid = []
            areas = []
            ann_ids = []
            for instance in instances:
                x1, y1, x2, y2 = [float(v) for v in instance["bbox_xyxy"]]
                boxes.append([x1 / width, y1 / height, x2 / width, y2 / height])

                keypoints = np.asarray(instance["keypoints"], dtype=np.float32).copy()
                if keypoints.shape != (17, 3):
                    raise ValueError(f"invalid keypoint shape: {keypoints.shape}")
                visible = keypoints[:, 2] > 0
                keypoints[visible, 0] /= width
                keypoints[visible, 1] /= height
                keypoints[~visible, :2] = 0.0
                poses.append(keypoints)
                pose_valid.append(bool(instance["pose_valid"]))

                current_mask = instance.get("mask")
                current_seg_valid = bool(
                    instance["seg_valid"] and current_mask is not None
                )
                masks.append(
                    torch.from_numpy(np.asarray(current_mask, dtype=np.float32))
                    if current_seg_valid
                    else None
                )
                seg_valid.append(current_seg_valid)
                genders.append(
                    int(instance["gender"]) if instance["gender_valid"] else -1
                )
                gender_valid.append(bool(instance["gender_valid"]))
                ages.append(
                    min(120, max(0, int(round(float(instance["age"])))))
                    if instance["age_valid"]
                    else -1
                )
                age_valid.append(bool(instance["age_valid"]))
                areas.append(float(instance["area"]))
                ann_ids.append(int(instance["source_ann_id"]))

            dense_targets.append(
                {
                    "boxes": torch.tensor(
                        boxes, dtype=torch.float32
                    ).reshape(-1, 4),
                    "poses": torch.tensor(
                        np.asarray(poses), dtype=torch.float32
                    ).reshape(-1, 17, 3),
                    "pose_valid": torch.tensor(pose_valid, dtype=torch.bool),
                    "masks": masks,
                    "seg_valid": torch.tensor(seg_valid, dtype=torch.bool),
                    "genders": torch.tensor(genders, dtype=torch.long),
                    "gender_valid": torch.tensor(gender_valid, dtype=torch.bool),
                    "ages": torch.tensor(ages, dtype=torch.long),
                    "age_valid": torch.tensor(age_valid, dtype=torch.bool),
                    "areas": torch.tensor(areas, dtype=torch.float32),
                    "ann_ids": ann_ids,
                    "image_size": (
                        int(sample["height"]),
                        int(sample["width"]),
                    ),
                    "person_count": len(instances),
                }
            )
        batch["dense_targets"] = dense_targets
        return batch
