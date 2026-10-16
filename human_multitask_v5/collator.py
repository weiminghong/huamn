from __future__ import annotations

from typing import Any, Dict, List, Sequence

import torch

from human_lisa_qwen_v4.schema_v4 import apply_chat_template_v4

from .protocol import HUMAN_TOKEN, PARSE_TOKEN, SEG_TOKEN


def find_last_subsequence(sequence: List[int], subsequence: List[int]) -> int:
    if not subsequence or len(subsequence) > len(sequence):
        return -1
    for start in range(len(sequence) - len(subsequence), -1, -1):
        if sequence[start : start + len(subsequence)] == subsequence:
            return start
    return -1


class QwenV5Collator:
    def __init__(
        self,
        *,
        processor: Any,
        token_ids: Dict[str, int],
        max_length: int = 5000,
    ) -> None:
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.token_ids = {key: int(value) for key, value in token_ids.items()}
        self.max_length = int(max_length)
        expected = {"human", "seg", "parse"}
        if set(self.token_ids) != expected:
            raise ValueError(f"token_ids must be {expected}")
        if self.tokenizer.padding_side != "right":
            raise ValueError("v5 training requires right padding")

    def prefix(self, prompt: str) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        return apply_chat_template_v4(self.processor, messages)

    def __call__(self, samples: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not samples:
            raise ValueError("empty batch")
        eos = int(self.tokenizer.eos_token_id)
        target_ids = [
            [
                int(value)
                for value in self.tokenizer(
                    sample["target_text"], add_special_tokens=False
                )["input_ids"]
            ]
            + [eos]
            for sample in samples
        ]
        texts = [
            self.prefix(sample["prompt"])
            + sample["target_text"]
            + str(self.tokenizer.eos_token)
            for sample in samples
        ]
        batch = self.processor(
            text=texts,
            images=[sample["image"] for sample in samples],
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        lengths = batch["attention_mask"].sum(1).long()
        labels = batch["input_ids"].clone()
        audits = []
        for row, (sample, target) in enumerate(zip(samples, target_ids)):
            length = int(lengths[row])
            if length > self.max_length:
                raise ValueError(
                    f"sequence overflow uid={sample['uid']} length={length}"
                )
            sequence = [
                int(value)
                for value in batch["input_ids"][row, :length].tolist()
            ]
            start = find_last_subsequence(sequence, target)
            if start < 0:
                raise RuntimeError(f"target tokens not found for {sample['uid']}")
            labels[row, :start] = -100
            human_count = target.count(self.token_ids["human"])
            seg_count = target.count(self.token_ids["seg"])
            parse_count = target.count(self.token_ids["parse"])
            if human_count != int(sample["person_count"]):
                raise RuntimeError(
                    f"human token count {human_count} != {sample['person_count']}"
                )
            dense = sample["dense"]
            if seg_count != int(dense["segment_owner_indices"].numel()):
                raise RuntimeError("segment route/target ownership mismatch")
            if parse_count != int(dense["parsing_owner_indices"].numel()):
                raise RuntimeError("parsing route/target ownership mismatch")
            audits.append(
                {
                    "uid": sample["uid"],
                    "task": sample["task"],
                    "persons": human_count,
                    "segments": seg_count,
                    "parsings": parse_count,
                    "tokens": length,
                }
            )
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        batch["sequence_lengths"] = lengths
        batch["sample_uids"] = [sample["uid"] for sample in samples]
        batch["tasks"] = [sample["task"] for sample in samples]
        batch["person_counts"] = [
            int(sample["person_count"]) for sample in samples
        ]
        batch["dense_targets"] = [sample["dense"] for sample in samples]
        batch["original_sizes"] = [
            sample["original_size"] for sample in samples
        ]
        batch["label_audit"] = audits
        return batch
