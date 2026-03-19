import os
from pathlib import Path
from typing import Any, Dict, Optional

from PIL import Image
from torch.utils.data import Dataset

from .io_utils import load_jsonl
from .prompting import build_stage1_prompt
from .target_builder import build_stage1_target


class Stage1HumanDataset(Dataset):
    """Stage1 VLM SFT dataset.

    Expected JSONL fields per sample:
      - id (optional)
      - image (required): relative/absolute image path
      - box / pose / parsing (optional for inference, required for training)
      - target_text (optional): if provided, used directly
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: Optional[str] = None,
        for_training: bool = True,
        max_samples: Optional[int] = None,
        extra_prompt: Optional[str] = None,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_root = image_root
        self.for_training = for_training
        self.extra_prompt = extra_prompt

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str) -> str:
        p = Path(image_path)
        if p.is_absolute():
            return str(p)
        if self.image_root is not None:
            return str(Path(self.image_root) / image_path)
        return str(p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_path = self._resolve_image_path(rec["image"])
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = build_stage1_prompt(self.extra_prompt)
        target_text = rec.get("target_text", build_stage1_target(rec))

        return {
            "id": rec.get("id", str(idx)),
            "image_path": image_path,
            "image": image,
            "prompt": prompt,
            "target_text": target_text,
            "labels": {
                "box": rec.get("box", []),
                "pose": rec.get("pose", []),
                "parsing": rec.get("parsing", []),
            },
            "for_training": self.for_training,
        }
