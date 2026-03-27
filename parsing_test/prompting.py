from typing import Optional


def build_qwen_parsing_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "You are a human parsing assistant.\n"
        "Return one JSON object only.\n"
        "Schema:\n"
        '{\n'
        '  "persons": [\n'
        "    {\n"
        '      "person_bbox": [x1, y1, x2, y2],\n'
        '      "parts": [\n'
        '        {"label": "hat", "bbox": [x1, y1, x2, y2]},\n'
        '        {"label": "hair", "bbox": [x1, y1, x2, y2]}\n'
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n"
        "Rules:\n"
        "1) Use only valid JSON, without markdown fences or explanations.\n"
        "2) Use pixel coordinates in the original image.\n"
        "3) Allowed part labels are: hat, hair, glove, sunglasses, upper_clothes, dress, coat, socks, pants, jumpsuits, scarf, skirt, face, left_arm, right_arm, left_leg, right_leg, left_shoe, right_shoe.\n"
        "4) If no visible people exist, return {\"persons\": []}.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
