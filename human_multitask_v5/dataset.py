from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from pycocotools import mask as mask_utils
from torch.utils.data import Dataset

from .manifest import iter_jsonl
from .protocol import PacketRef, build_target, task_prompt


def decode_binary_mask(
    label: Mapping[str, Any], height: int, width: int
) -> torch.Tensor:
    encoding = label["encoding"]
    kind = encoding["type"]
    if kind == "polygons":
        rles = mask_utils.frPyObjects(
            encoding["data"], int(height), int(width)
        )
        value = mask_utils.decode(mask_utils.merge(rles))
    elif kind == "coco_rle":
        rle = encoding["data"]
        if isinstance(rle.get("counts"), list):
            rle = mask_utils.frPyObjects(rle, int(height), int(width))
        value = mask_utils.decode(rle)
    elif kind == "png":
        with Image.open(encoding["path"]) as source:
            value = np.asarray(source.convert("L")) > 0
    else:
        raise ValueError(f"not a binary mask encoding: {kind}")
    if value.ndim == 3:
        value = np.any(value, axis=2)
    if value.shape != (height, width):
        raise ValueError(
            f"decoded mask shape {value.shape} != {(height, width)}"
        )
    return torch.from_numpy(np.asarray(value, dtype=np.float32))


def decode_parsing(
    label: Mapping[str, Any], height: int, width: int
) -> torch.Tensor:
    encoding = label["encoding"]
    if encoding["type"] != "label_png":
        raise ValueError("parsing currently requires official label_png")
    with Image.open(encoding["path"]) as source:
        value = np.asarray(source, dtype=np.uint8)
    if value.shape != (height, width):
        raise ValueError(
            f"decoded parsing shape {value.shape} != {(height, width)}"
        )
    if value.min() < 0 or value.max() > 58:
        raise ValueError("MHP parsing labels must be in [0,58]")
    return torch.from_numpy(value.astype(np.int64))


class CanonicalPacketDataset(Dataset):
    def __init__(
        self,
        manifests: Sequence[str],
        packets: Sequence[PacketRef],
    ) -> None:
        self.records: List[Dict] = [
            record for path in manifests for record in iter_jsonl(path)
        ]
        self.packets = list(packets)
        if not self.records or not self.packets:
            raise ValueError("records and packets must be non-empty")

    def __len__(self) -> int:
        return len(self.packets)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        packet = self.packets[index]
        record = self.records[packet.record_index]
        image_path = record["image"]["path"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        width, height = image.size
        if (width, height) != (
            int(record["image"]["width"]),
            int(record["image"]["height"]),
        ):
            raise ValueError("manifest and image dimensions differ")
        dense = {
            "boxes": [],
            "segment_owner_indices": [],
            "segment_masks": [],
            "parsing_owner_indices": [],
            "parsing_maps": [],
        }
        for owner_index, person in enumerate(record["persons"]):
            x1, y1, x2, y2 = person["bbox_xyxy"]
            dense["boxes"].append(
                [x1 / width, y1 / height, x2 / width, y2 / height]
            )
            labels = person["labels"]
            if (
                packet.task in ("segment", "all")
                and labels["segment"].get("valid", False)
            ):
                dense["segment_owner_indices"].append(owner_index)
                dense["segment_masks"].append(
                    decode_binary_mask(labels["segment"], height, width)
                )
            if (
                packet.task in ("parsing", "all")
                and labels["parsing"].get("valid", False)
            ):
                dense["parsing_owner_indices"].append(owner_index)
                dense["parsing_maps"].append(
                    decode_parsing(labels["parsing"], height, width)
                )
        dense["boxes"] = torch.tensor(
            dense["boxes"], dtype=torch.float32
        ).reshape(-1, 4)
        dense["segment_owner_indices"] = torch.tensor(
            dense["segment_owner_indices"], dtype=torch.long
        )
        dense["parsing_owner_indices"] = torch.tensor(
            dense["parsing_owner_indices"], dtype=torch.long
        )
        return {
            "uid": record["uid"],
            "dataset": record["dataset"],
            "task": packet.task,
            "image": image,
            "prompt": task_prompt(packet.task),
            "target_text": build_target(record, packet.task),
            "person_count": len(record["persons"]),
            "dense": dense,
            "original_size": (height, width),
        }
