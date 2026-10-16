from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .targets_v1 import SerializedTarget, serialize_target


class SequenceOverflowError(RuntimeError):
    pass


def _find_last_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    first = subsequence[0]
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start] != first:
            continue
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


def _overlapping_token_indices(
    offsets: Sequence[Tuple[int, int]],
    spans: Sequence[Tuple[int, int, str, int]],
) -> List[int]:
    result = set()
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        for span_start, span_end, _, _ in spans:
            if token_start < span_end and token_end > span_start:
                result.add(token_index)
    return sorted(result)


class Stage1Collator:
    def __init__(
        self,
        *,
        processor: Any,
        token_ids: Dict[str, int],
        prompt: str,
        max_length: int = 5000,
        missing_attribute_loss_weight: float = 0.0,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.token_ids = {key: int(value) for key, value in token_ids.items()}
        self.prompt = str(prompt)
        self.max_length = int(max_length)
        self.missing_attribute_loss_weight = float(
            missing_attribute_loss_weight
        )
        if not 0.0 <= self.missing_attribute_loss_weight <= 1.0:
            raise ValueError(
                "missing attribute loss weight must be in [0, 1]"
            )
        if self.max_length <= 0:
            raise ValueError("max_length must be positive")
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("Stage1 training requires right padding")

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
        return self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _target_encoding(
        self, target: SerializedTarget
    ) -> Tuple[str, List[int], List[Tuple[int, int]], List[int]]:
        eos = self.tokenizer.eos_token or ""
        target_with_eos = target.text + eos
        encoded = self.tokenizer(
            target_with_eos,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        target_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        missing = _overlapping_token_indices(
            offsets, target.missing_attribute_char_spans
        )
        return target_with_eos, target_ids, offsets, missing

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("collator received an empty batch")
        prefix = self._chat_prefix()
        targets = [
            serialize_target(
                sample["instances"],
                width=int(sample["width"]),
                height=int(sample["height"]),
            )
            for sample in samples
        ]
        target_encodings = [
            self._target_encoding(target) for target in targets
        ]
        texts = [
            prefix + target_encoding[0]
            for target_encoding in target_encodings
        ]
        images = [sample["image"] for sample in samples]
        batch = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        attention_mask = batch["attention_mask"]
        sequence_lengths = attention_mask.sum(dim=1).to(torch.long)
        labels = batch["input_ids"].clone()
        language_loss_weights = torch.ones_like(labels, dtype=torch.float32)
        audits: List[Dict[str, Any]] = []

        for row, (sample, target, encoded) in enumerate(
            zip(samples, targets, target_encodings)
        ):
            _, target_ids, _, missing_target_positions = encoded
            sequence_length = int(sequence_lengths[row].item())
            if sequence_length > self.max_length:
                raise SequenceOverflowError(
                    "truncation is disabled: "
                    f"image_id={sample['id']} "
                    f"persons={len(sample['instances'])} "
                    f"target_tokens={len(target_ids)} "
                    f"total_tokens={sequence_length} "
                    f"max_length={self.max_length}"
                )
            sequence_ids = [
                int(value)
                for value in batch["input_ids"][row, :sequence_length].tolist()
            ]
            target_start = _find_last_subsequence(sequence_ids, target_ids)
            if target_start < 0:
                raise RuntimeError(
                    f"cannot locate assistant target tokens for image_id={sample['id']}"
                )
            language_loss_weights[row, :target_start] = 0.0
            labels[row, :target_start] = -100
            absolute_missing = [
                target_start + position for position in missing_target_positions
            ]
            for position in absolute_missing:
                labels[row, position] = -100
                language_loss_weights[row, position] = 0.0

            special_counts = {
                name: target_ids.count(token_id)
                for name, token_id in self.token_ids.items()
            }
            if special_counts != target.target_counts:
                raise RuntimeError(
                    "special-token/target mismatch: "
                    f"image_id={sample['id']} "
                    f"tokens={special_counts} targets={target.target_counts}"
                )
            audits.append(
                {
                    "image_id": int(sample["id"]),
                    "persons": len(sample["instances"]),
                    "target_start": target_start,
                    "target_tokens": len(target_ids),
                    "total_tokens": sequence_length,
                    "masked_attribute_token_positions": absolute_missing,
                    "special_token_counts": special_counts,
                    "target_counts": target.target_counts,
                }
            )

        labels[attention_mask == 0] = -100
        language_loss_weights[attention_mask == 0] = 0.0
        language_loss_weights[labels == -100] = 0.0
        batch["labels"] = labels
        batch["language_loss_weights"] = language_loss_weights
        batch["sequence_lengths"] = sequence_lengths
        batch["label_audit"] = audits
        batch["sample_ids"] = [int(sample["id"]) for sample in samples]
        batch["target_texts"] = [target.text for target in targets]
        batch["source_ann_ids"] = [target.source_ann_ids for target in targets]
        batch["box_targets"] = [
            torch.from_numpy(target.box_targets) for target in targets
        ]
        batch["pose_targets"] = [
            torch.from_numpy(target.pose_targets) for target in targets
        ]
        batch["pose_areas"] = [
            torch.from_numpy(target.pose_areas) for target in targets
        ]
        batch["seg_targets"] = [
            torch.from_numpy(
                (
                    np.stack(target.seg_targets)
                    if target.seg_targets
                    else np.zeros(
                        (0, int(sample["height"]), int(sample["width"])),
                        dtype=bool,
                    )
                )
            )
            for sample, target in zip(samples, targets)
        ]
        batch["box_person_indices"] = [
            torch.tensor(target.box_person_indices, dtype=torch.long)
            for target in targets
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

