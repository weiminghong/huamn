from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .schema_v4 import apply_chat_template_v4
from .targets_v4 import (
    IGNORE_TASK_ID,
    TEXT_TASK_TO_ID,
    SerializedTargetV4,
    assign_token_tasks,
    serialize_target_v4,
)


class SequenceOverflowError(RuntimeError):
    pass


def _find_last_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    first = subsequence[0]
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start] == first and sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


class Stage1V4Collator:
    def __init__(
        self,
        *,
        processor: Any,
        token_ids: Dict[str, int],
        prompt: str,
        max_length: int = 5000,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.token_ids = {name: int(value) for name, value in token_ids.items()}
        self.prompt = str(prompt)
        self.max_length = int(max_length)
        if self.max_length != 5000:
            raise ValueError("v4 requires max_length=5000")
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("v4 training requires right padding")
        if self.tokenizer.eos_token_id is None:
            raise ValueError("tokenizer must define eos_token_id")

    def _chat_prefix(self) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]
        return apply_chat_template_v4(self.processor, messages)

    def _target_encoding(
        self, target: SerializedTargetV4
    ) -> Tuple[str, List[int], List[int], List[bool]]:
        encoded = self.tokenizer(
            target.text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        target_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        task_ids, valid_flags = assign_token_tasks(offsets, target.spans)
        target_ids.append(int(self.tokenizer.eos_token_id))
        task_ids.append(TEXT_TASK_TO_ID["structure"])
        valid_flags.append(True)
        target_with_eos = target.text + str(self.tokenizer.eos_token)
        check = self.tokenizer(target_with_eos, add_special_tokens=False)["input_ids"]
        if [int(value) for value in check] != target_ids:
            raise RuntimeError("target plus EOS does not tokenize compositionally")
        return target_with_eos, target_ids, task_ids, valid_flags

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("collator received an empty batch")
        prefix = self._chat_prefix()
        targets = [
            serialize_target_v4(
                sample["instances"],
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
            for sample in samples
        ]
        encodings = [self._target_encoding(target) for target in targets]
        texts = [prefix + encoding[0] for encoding in encodings]
        batch = self.processor(
            text=texts,
            images=[sample["image"] for sample in samples],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        attention_mask = batch["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1).to(torch.long)
        labels = batch["input_ids"].clone()
        text_task_ids = torch.full_like(labels, IGNORE_TASK_ID)
        audits: List[Dict[str, Any]] = []

        for row, (sample, target, encoding) in enumerate(
            zip(samples, targets, encodings)
        ):
            _, target_ids, target_task_ids, valid_flags = encoding
            sequence_length = int(sequence_lengths[row].item())
            if sequence_length > self.max_length:
                raise SequenceOverflowError(
                    "truncation is disabled: "
                    f"image_id={sample['id']} persons={target.person_count} "
                    f"target_tokens={len(target_ids)} total_tokens={sequence_length} "
                    f"max_length={self.max_length}"
                )
            sequence_ids = [
                int(value)
                for value in batch["input_ids"][row, :sequence_length].tolist()
            ]
            target_start = _find_last_subsequence(sequence_ids, target_ids)
            if target_start < 0:
                raise RuntimeError(
                    f"cannot locate assistant target for image_id={sample['id']}"
                )
            labels[row, :target_start] = -100
            for offset, (task_id, is_valid) in enumerate(
                zip(target_task_ids, valid_flags)
            ):
                position = target_start + offset
                if is_valid:
                    text_task_ids[row, position] = int(task_id)
                else:
                    labels[row, position] = -100
            pose_count = target_ids.count(self.token_ids["pose"])
            seg_count = target_ids.count(self.token_ids["seg"])
            if pose_count != target.person_count or seg_count != target.person_count:
                raise RuntimeError(
                    "routing-token/person mismatch: "
                    f"image_id={sample['id']} persons={target.person_count} "
                    f"pose={pose_count} seg={seg_count}"
                )
            audits.append(
                {
                    "image_id": int(sample["id"]),
                    "persons": target.person_count,
                    "target_start": target_start,
                    "target_tokens": len(target_ids),
                    "total_tokens": sequence_length,
                    "pose_labels": len(target.pose_person_indices),
                    "seg_labels": len(target.seg_person_indices),
                    "task_token_counts": {
                        name: target_task_ids.count(task_id)
                        for name, task_id in TEXT_TASK_TO_ID.items()
                    },
                }
            )

        labels[attention_mask == 0] = -100
        text_task_ids[labels == -100] = IGNORE_TASK_ID
        batch["labels"] = labels
        batch["text_task_ids"] = text_task_ids
        batch["sequence_lengths"] = sequence_lengths
        batch["label_audit"] = audits
        batch["sample_ids"] = [int(sample["id"]) for sample in samples]
        batch["target_texts"] = [target.text for target in targets]
        batch["source_ann_ids"] = [target.source_ann_ids for target in targets]
        batch["person_counts"] = [target.person_count for target in targets]
        batch["pose_targets"] = [
            torch.from_numpy(target.pose_targets) for target in targets
        ]
        batch["pose_areas"] = [
            torch.from_numpy(target.pose_areas) for target in targets
        ]
        batch["seg_targets"] = [
            torch.from_numpy(
                np.stack(target.seg_targets)
                if target.seg_targets
                else np.zeros(
                    (0, int(sample["height"]), int(sample["width"])),
                    dtype=bool,
                )
            )
            for sample, target in zip(samples, targets)
        ]
        batch["pose_person_indices"] = [
            torch.tensor(target.pose_person_indices, dtype=torch.long)
            for target in targets
        ]
        batch["seg_person_indices"] = [
            torch.tensor(target.seg_person_indices, dtype=torch.long)
            for target in targets
        ]
        batch["original_sizes"] = [
            (int(sample["height"]), int(sample["width"])) for sample in samples
        ]
        return batch
