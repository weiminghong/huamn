import os
from typing import Dict, List, Optional

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


class LipParsingDataset(Dataset):
    def __init__(
        self,
        image_dir: str,
        ann_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.items = self._scan_items()
        if max_samples is not None:
            self.items = self.items[: max_samples]

    def _scan_items(self) -> List[Dict[str, Optional[str]]]:
        if not os.path.isdir(self.image_dir):
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        names = []
        for fn in sorted(os.listdir(self.image_dir)):
            if fn.lower().endswith(IMAGE_EXTS):
                names.append(fn)

        items: List[Dict[str, Optional[str]]] = []
        for fn in names:
            stem = os.path.splitext(fn)[0]
            ann_path = None
            if self.ann_dir is not None:
                ann_candidate = os.path.join(self.ann_dir, f"{stem}.png")
                if os.path.isfile(ann_candidate):
                    ann_path = ann_candidate
            items.append(
                {
                    "id": stem,
                    "image_path": os.path.join(self.image_dir, fn),
                    "ann_path": ann_path,
                }
            )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Dict[str, object]:
        item = self.items[index]
        image = Image.open(item["image_path"]).convert("RGB")
        gt_mask = None
        if item["ann_path"] is not None:
            gt_mask = np.array(Image.open(item["ann_path"]), dtype=np.int64)
        return {
            "id": item["id"],
            "image_path": item["image_path"],
            "image": image,
            "gt_mask": gt_mask,
        }
