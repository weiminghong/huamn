from typing import Optional


def build_stage1_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "Output structured human annotations in a single JSON object only.\n"
        "Required top-level fields:\n"
        '- "box": list of person boxes, each item should include "bbox_xyxy".\n'
        '- "pose": list of person keypoints, each item should include "keypoints".\n'
        '- "parsing": list of human parsing items, each item should include "labels" or "segments".\n'
        "Rules:\n"
        "1) Always include all three fields; use empty lists when missing.\n"
        "2) Use pixel coordinates when possible.\n"
        "3) Return valid JSON only. No extra explanation.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base


def build_stage1_five_task_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "You are a human-centric perception model.\n"
        "Return one JSON object only with these top-level fields:\n"
        '- "box": person boxes, each item uses "bbox_xyxy".\n'
        '- "pose": person keypoints, each item uses "keypoints".\n'
        '- "attribute": person attributes as key-value dictionaries.\n'
        '- "relation": relation triples among person and objects.\n'
        '- "parsing": each item must include a segmentation token "<SEG>".\n'
        "Rules:\n"
        "1) Keep valid JSON only, no explanation text.\n"
        "2) Keep all five fields even when empty.\n"
        "3) Number of <SEG> tokens in parsing must match provided mask targets.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
