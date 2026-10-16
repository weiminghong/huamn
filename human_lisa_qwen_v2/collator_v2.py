from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from .targets_v2 import CharSpan, SerializedTargetV2, serialize_target_v2


CE_TASK_IDS = {"structure": 0, "bbox": 1, "gender": 2, "age": 3, "anchor": 4}


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


def _overlapping_token_indices(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[CharSpan]
) -> List[int]:
    result = set()
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        for span_start, span_end, _, _ in spans:
            if token_start < span_end and token_end > span_start:
                result.add(token_index)
    return sorted(result)


def _task_ids_from_offsets(
    offsets: Sequence[Tuple[int, int]], spans: Sequence[CharSpan]
) -> List[int]:
    result = [CE_TASK_IDS["structure"] for _ in offsets]
    for token_index, (token_start, token_end) in enumerate(offsets):
        if token_end <= token_start:
            continue
        matching = {
            task
            for span_start, span_end, task, _ in spans
            if token_start < span_end and token_end > span_start
        }
        if len(matching) > 1:
            raise RuntimeError(f"ambiguous CE task span at token {token_index}: {matching}")
        if matching:
            result[token_index] = CE_TASK_IDS[next(iter(matching))]
    return result


class Stage1V2Collator:
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
        self.token_ids = {key: int(value) for key, value in token_ids.items()}
        self.prompt = str(prompt)
        self.max_length = int(max_length)
        if self.max_length != 5000:
            raise ValueError("Stage1 V2 uses a fixed audited max_length=5000")
        if getattr(self.tokenizer, "padding_side", "right") != "right":
            raise ValueError("Stage1 V2 training requires right padding")

    def _chat_prefix(self) -> str:
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]
        return self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _target_encoding(
        self, target: SerializedTargetV2
    ) -> Tuple[str, List[int], List[Tuple[int, int]], List[int], List[int]]:
        eos = self.tokenizer.eos_token or ""
        target_with_eos = target.text + eos
        encoded = self.tokenizer(
            target_with_eos,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        target_ids = [int(value) for value in encoded["input_ids"]]
        offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"]]
        missing = _overlapping_token_indices(offsets, target.missing_attribute_char_spans)
        task_ids = _task_ids_from_offsets(offsets, target.ce_task_char_spans)
        return target_with_eos, target_ids, offsets, missing, task_ids

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("collator received an empty batch")
        prefix = self._chat_prefix()
        targets = [
            serialize_target_v2(sample["instances"], width=int(sample["width"]), height=int(sample["height"]))
            for sample in samples
        ]
        target_encodings = [self._target_encoding(target) for target in targets]
        texts = [prefix + encoded[0] for encoded in target_encodings]
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
        language_loss_weights = torch.ones_like(labels, dtype=torch.float32)
        language_task_ids = torch.full_like(labels, CE_TASK_IDS["structure"])
        audits: List[Dict[str, Any]] = []

        for row, (sample, target, encoded) in enumerate(zip(samples, targets, target_encodings)):
            _, target_ids, _, missing_positions, task_ids = encoded
            sequence_length = int(sequence_lengths[row].item())
            if sequence_length > self.max_length:
                raise SequenceOverflowError(
                    "truncation is disabled: "
                    f"image_id={sample['id']} persons={target.num_persons} "
                    f"target_tokens={len(target_ids)} total_tokens={sequence_length} "
                    f"max_length={self.max_length}"
                )
            sequence_ids = [int(value) for value in batch["input_ids"][row, :sequence_length].tolist()]
            target_start = _find_last_subsequence(sequence_ids, target_ids)
            if target_start < 0:
                raise RuntimeError(f"cannot locate assistant target for image_id={sample['id']}")
            labels[row, :target_start] = -100
            language_loss_weights[row, :target_start] = 0.0
            language_task_ids[row, target_start : target_start + len(task_ids)] = torch.tensor(task_ids, dtype=torch.long)
            absolute_missing = [target_start + position for position in missing_positions]
            for position in absolute_missing:
                labels[row, position] = -100
                language_loss_weights[row, position] = 0.0
            special_counts = {
                name: target_ids.count(token_id) for name, token_id in self.token_ids.items()
            }
            if special_counts != target.target_counts:
                raise RuntimeError(
                    "special-token/record mismatch: "
                    f"image_id={sample['id']} tokens={special_counts} targets={target.target_counts}"
                )
            audits.append(
                {
                    "image_id": int(sample["id"]),
                    "persons": target.num_persons,
                    "target_start": target_start,
                    "target_tokens": len(target_ids),
                    "total_tokens": sequence_length,
                    "masked_attribute_token_positions": absolute_missing,
                    "special_token_counts": special_counts,
                    "valid_pose": len(target.pose_person_indices),
                    "valid_seg": len(target.seg_person_indices),
                }
            )

        labels[attention_mask == 0] = -100
        language_loss_weights[attention_mask == 0] = 0.0
        language_loss_weights[labels == -100] = 0.0
        batch["labels"] = labels
        batch["language_loss_weights"] = language_loss_weights
        batch["language_task_ids"] = language_task_ids
        batch["sequence_lengths"] = sequence_lengths
        batch["label_audit"] = audits
        batch["sample_ids"] = [int(sample["id"]) for sample in samples]
        batch["target_texts"] = [target.text for target in targets]
        batch["source_ann_ids"] = [target.source_ann_ids for target in targets]
        batch["num_persons"] = [target.num_persons for target in targets]
        batch["box_targets"] = [torch.from_numpy(target.box_targets) for target in targets]
        batch["pose_targets"] = [torch.from_numpy(target.pose_targets) for target in targets]
        batch["pose_areas"] = [torch.from_numpy(target.pose_areas) for target in targets]
        batch["seg_targets"] = [
            torch.from_numpy(
                np.stack(target.seg_targets)
                if target.seg_targets
                else np.zeros((0, int(sample["height"]), int(sample["width"])), dtype=bool)
            )
            for sample, target in zip(samples, targets)
        ]
        batch["pose_person_indices"] = [torch.tensor(target.pose_person_indices, dtype=torch.long) for target in targets]
        batch["seg_person_indices"] = [torch.tensor(target.seg_person_indices, dtype=torch.long) for target in targets]
        batch["original_sizes"] = [(int(sample["height"]), int(sample["width"])) for sample in samples]
        return batch
