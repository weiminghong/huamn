from __future__ import annotations

import re
from typing import Tuple

from qwen4b_coco_unihuman_ft.eval_scheme2 import parse_tag_struct

TRIPLET_RE = re.compile(
    r"\[\s*-?(?:\d+\.\d+|\d+|\.\d+)\s*,\s*-?(?:\d+\.\d+|\d+|\.\d+)\s*,\s*-?(?:\d+\.\d+|\d+|\.\d+)\s*\]"
)


def contains_think_tokens(text: str) -> bool:
    if not isinstance(text, str):
        return False
    return bool(re.search(r"</?think>", text, flags=re.IGNORECASE))


def answer_block_count(text: str) -> int:
    if not isinstance(text, str):
        return 0
    return len(re.findall(r"<answer>", text, flags=re.IGNORECASE))


def truncate_to_first_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    start_match = re.search(r"<answer>", text, flags=re.IGNORECASE)
    if start_match:
        text = text[start_match.start() :]
    end_match = re.search(r"</answer>", text, flags=re.IGNORECASE)
    if end_match:
        text = text[: end_match.end()]
    text = re.sub(r"</?think>", "", text, flags=re.IGNORECASE)
    return text.strip()


def _has_strict_pose_triplets(text: str) -> bool:
    pose_blobs = re.findall(r"<pose>(.*?)</pose>", text, flags=re.IGNORECASE | re.DOTALL)
    if not pose_blobs:
        return False
    for blob in pose_blobs:
        if len(TRIPLET_RE.findall(blob)) != 17:
            return False
    return True


def prediction_needs_repair(text: str, image_size: Tuple[int, int]) -> tuple[bool, str]:
    cleaned = truncate_to_first_answer(text)
    if contains_think_tokens(text):
        return True, cleaned
    if answer_block_count(cleaned) != 1:
        return True, cleaned
    if not _has_strict_pose_triplets(cleaned):
        return True, cleaned
    if parse_tag_struct(cleaned, image_size=image_size) is None:
        return True, cleaned
    return False, cleaned
