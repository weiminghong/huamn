import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from torch.utils.data import Dataset

from .prompting import build_box_prompt
from .target_builder import build_box_target, xywh_to_xyxy


IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def _find_image_for_stem(image_dir: str, stem: str) -> Optional[str]:
    for ext in IMAGE_EXTS:
        path = osp.join(image_dir, stem + ext)
        if osp.isfile(path):
            return path
    return None


class BoxTxtDataset(Dataset):
    """Image + paired txt dataset for box-only structured generation.

    Label txt file:
      one object per line
      x y w h
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        list_file: Optional[str] = None,
        for_training: bool = True,
        max_samples: Optional[int] = None,
        extra_prompt: Optional[str] = None,
    ) -> None:
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.for_training = for_training
        self.extra_prompt = extra_prompt
        self.records = self._build_records(list_file)
        if max_samples is not None:
            self.records = self.records[:max_samples]

    def _build_records(self, list_file: Optional[str]) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        stems: List[str] = []

        if list_file is not None:
            with open(list_file, "r", encoding="utf-8") as f:
                for line in f:
                    name = line.strip()
                    if not name:
                        continue
                    stems.append(Path(name).stem)
        else:
            if not osp.isdir(self.label_dir):
                raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
            for fn in sorted(os.listdir(self.label_dir)):
                if fn.lower().endswith(".txt"):
                    stems.append(Path(fn).stem)

        for stem in stems:
            image_path = _find_image_for_stem(self.image_dir, stem)
            if image_path is None:
                continue
            label_path = osp.join(self.label_dir, stem + ".txt")
            if not osp.isfile(label_path):
                continue
            boxes = self._load_label_file(label_path)
            records.append(
                {
                    "id": stem,
                    "image_path": image_path,
                    "label_path": label_path,
                    "box": boxes,
                }
            )
        return records

    def _load_label_file(self, path: str) -> List[Dict[str, List[float]]]:
        boxes: List[Dict[str, List[float]]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 4:
                    continue
                x, y, w, h = [float(v) for v in parts[:4]]
                boxes.append({"bbox_xyxy": xywh_to_xyxy([x, y, w, h])})
        return boxes

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image = Image.open(rec["image_path"]).convert("RGB")
        prompt = build_box_prompt(self.extra_prompt)
        target_text = build_box_target(rec)
        return {
            "id": rec["id"],
            "image_path": rec["image_path"],
            "label_path": rec["label_path"],
            "image": image,
            "prompt": prompt,
            "target_text": target_text,
            "labels": {"box": rec["box"]},
            "for_training": self.for_training,
        }
