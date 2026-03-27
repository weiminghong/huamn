from typing import Optional


def build_box_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "Detect all visible persons and return one JSON object only.\n"
        "Schema:\n"
        '{"box":[{"bbox_xyxy":[x1,y1,x2,y2],"score":0.99}]}\n'
        "Rules:\n"
        '1) Keep the top-level field "box" even if empty.\n'
        "2) Use pixel coordinates in the original image.\n"
        "3) Each item must contain bbox_xyxy. score is optional.\n"
        "4) Return valid JSON only. No markdown fences or explanation.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
