import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .five_task_builder import build_five_task_target_text
from .io_utils import load_jsonl
from .prompting import build_stage1_five_task_prompt


class Stage1FiveTaskDataset(Dataset):
    """Stage1 five-task VLM dataset with parsing masks."""

    def __init__(
        self,
        jsonl_path: str,
        image_root: Optional[str] = None,
        mask_root: Optional[str] = None,
        for_training: bool = True,
        max_samples: Optional[int] = None,
        extra_prompt: Optional[str] = None,
        seg_token: str = "<SEG>",
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_root = image_root
        self.mask_root = mask_root
        self.for_training = for_training
        self.extra_prompt = extra_prompt
        self.seg_token = seg_token

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_path(self, raw_path: str, root: Optional[str]) -> str:
        p = Path(raw_path)
        if p.is_absolute():
            return str(p)
        if root is not None:
            return str(Path(root) / raw_path)
        return str(p)

    def _load_mask_tensor(self, mask_path: str) -> torch.Tensor:
        with Image.open(mask_path) as im:
            arr = np.array(im.convert("L"), dtype=np.uint8)
        binary = (arr > 0).astype(np.float32)
        return torch.from_numpy(binary)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_path = self._resolve_path(rec["image"], self.image_root)
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = rec.get("prompt", build_stage1_five_task_prompt(self.extra_prompt))
        target_text = rec.get("target_text")

        if not target_text:
            target_text, default_masks, payload = build_five_task_target_text(
                rec,
                seg_token=self.seg_token,
            )
            labels = payload
            mask_paths = rec.get("mask_paths", default_masks)
        else:
            labels = {
                "box": rec.get("box", []),
                "pose": rec.get("pose", []),
                "attribute": rec.get("attribute", []),
                "relation": rec.get("relation", []),
                "parsing": rec.get("parsing", []),
            }
            mask_paths = rec.get("mask_paths", [])

        resolved_masks: List[str] = []
        mask_tensors: List[torch.Tensor] = []
        for path in mask_paths:
            if not isinstance(path, str) or not path:
                continue
            resolved = self._resolve_path(path, self.mask_root)
            if not os.path.isfile(resolved):
                continue
            resolved_masks.append(resolved)
            mask_tensors.append(self._load_mask_tensor(resolved))

        return {
            "id": rec.get("id", str(idx)),
            "image_path": image_path,
            "image": image,
            "prompt": prompt,
            "target_text": target_text,
            "mask_paths": resolved_masks,
            "mask_tensors": mask_tensors,
            "labels": labels,
            "for_training": self.for_training,
        }
