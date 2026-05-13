from pathlib import Path
from typing import Optional

from .jsonl_dataset import HumanJsonlDataset
from .raw_unihuman_dataset import RawCocoUniHumanDataset, human_collate


def build_dataset(
    annotation_path: str,
    image_root: str,
    max_samples: Optional[int] = None,
):
    if Path(annotation_path).suffix.lower() == ".jsonl":
        return HumanJsonlDataset(annotation_path, image_root=image_root, max_samples=max_samples)
    return RawCocoUniHumanDataset(annotation_path, image_root=image_root, max_samples=max_samples)


__all__ = ["HumanJsonlDataset", "RawCocoUniHumanDataset", "build_dataset", "human_collate"]
