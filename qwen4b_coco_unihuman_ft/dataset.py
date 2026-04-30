import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from .convert_coco_unihuman import _build_target_text
from .prompts import (
    DEFAULT_ASSISTANT_PREFIX_TAG,
    build_box_pose_attribute_prompt,
    build_vl_chat_text,
    strip_answer_prefix,
)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class CocoUniHumanSFTDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        image_root: Optional[str] = None,
        max_samples: Optional[int] = None,
        extra_prompt: Optional[str] = None,
    ) -> None:
        self.records = load_jsonl(jsonl_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self.image_root = image_root
        self.extra_prompt = extra_prompt

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str) -> str:
        p = Path(image_path)
        if p.is_absolute():
            return str(p)
        if self.image_root:
            return str(Path(self.image_root) / image_path)
        return str(p)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rec = self.records[idx]
        image_path = self._resolve_image_path(str(rec.get("image_abs", rec["image"])))
        if not os.path.isfile(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        prompt = build_box_pose_attribute_prompt(self.extra_prompt)
        target_text = str(rec.get("target_text", "")).strip()
        if not target_text or not target_text.lower().startswith("<answer"):
            target_text = _build_target_text(rec)

        return {
            "id": rec.get("id", str(idx)),
            "image": image,
            "image_path": image_path,
            "prompt": prompt,
            "target_text": target_text,
        }


class SFTCollator:
    def __init__(self, processor, tokenizer, max_length: int) -> None:
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.assistant_prefix = DEFAULT_ASSISTANT_PREFIX_TAG

    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        prompts = [str(s["prompt"]) for s in samples]
        targets = [strip_answer_prefix(str(s["target_text"])) for s in samples]
        prefixes = [
            build_vl_chat_text(
                processor=self.processor,
                prompt=prompt,
                assistant_prefix=self.assistant_prefix,
            )
            for prompt in prompts
        ]
        texts = [p + t for p, t in zip(prefixes, targets)]
        images = [s["image"] for s in samples]

        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        labels = batch["input_ids"].clone()
        prefix_lens = [
            len(self.tokenizer(p, add_special_tokens=False).input_ids) for p in prefixes
        ]
        for i, plen in enumerate(prefix_lens):
            labels[i, : min(plen, labels.shape[1])] = -100
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch
