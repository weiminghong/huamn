from typing import Optional

from .output_protocol import FORMAT_COMPACT_TAG, FORMAT_TAG, normalize_target_format

DEFAULT_ASSISTANT_PREFIX_TAG = "<answer>\n"


COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)


PROMPT_BOX_POSE_ATTR_TAG = (
    "<image>\n"
    "You are a human-structure annotation model.\n\n"
    "Task:\n"
    "Detect each distinct visible real person in the image and output box, COCO-17 pose, gender, age, and score.\n\n"
    "Output contract:\n"
    "Return exactly one XML block:\n"
    "<answer>\n"
    "... zero or more <person> blocks ...\n"
    "</answer>\n\n"
    "Allowed tags:\n"
    "<answer>, </answer>, <person>, </person>, <box>, </box>, <pose>, </pose>, <gender>, </gender>, <age>, </age>, <score>, </score>\n\n"
    "Do not output anything else. No markdown. No JSON. No bullets. No explanations. No comments. No XML attributes. No chain-of-thought. Stop immediately after </answer>.\n\n"
    "Person selection:\n"
    "- Output every clear visible real person exactly once.\n"
    "- Include partially visible people if a person instance is clear.\n"
    "- Skip reflections, posters, statues, mannequins, cropped body parts without a clear person instance, and ambiguous tiny regions.\n"
    "- Do not duplicate one person with multiple nearby boxes.\n"
    "- Output persons in left-to-right order by box center; break ties top-to-bottom.\n"
    "- Do not use a fixed maximum person count.\n\n"
    "Coordinate system:\n"
    "- All coordinates are integer values normalized to [0,1000].\n"
    "- x increases left to right. y increases top to bottom.\n"
    "- <box> is [x1, y1, x2, y2], a tight box around the visible person extent.\n"
    "- Require 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.\n\n"
    "Pose:\n"
    "- <pose> must contain exactly 17 triplets in this COCO order:\n"
    "  "
    f"{COCO_KEYPOINT_ORDER}.\n"
    "- Each triplet is [x, y, v].\n"
    "- v=2 means visible. v=1 means occluded but inferable. v=0 means missing, outside image, or not reliably inferable.\n"
    "- If v=0, output exactly [0, 0, 0].\n"
    "- If v is 1 or 2, x and y must be normalized integers in [0,1000].\n"
    "- Output exactly 17 triplets, no more and no fewer.\n\n"
    "Attributes:\n"
    "- <gender> is 0 for male, 1 for female, -1 for unknown.\n"
    "- <age> is estimated age in years, or -1 for unknown.\n"
    "- <score> is detection confidence in [0,1]. If score is not calibrated by training, use 1.0 for each emitted person.\n\n"
    "Per-person numeric form. Follow this tag structure and replace the example numbers with your predictions:\n"
    "<person>\n"
    "<box>[120, 250, 480, 820]</box>\n"
    "<pose>[250, 300, 2], [260, 280, 2], [240, 280, 2], [280, 290, 2], [220, 290, 2], [300, 350, 2], [200, 350, 2], [350, 450, 2], [150, 450, 2], [380, 500, 1], [120, 500, 2], [280, 600, 2], [220, 600, 2], [290, 750, 1], [210, 750, 1], [0, 0, 0], [0, 0, 0]</pose>\n"
    "<gender>0</gender>\n"
    "<age>25</age>\n"
    "<score>1.0</score>\n"
    "</person>\n\n"
    "Numeric fields must contain digits only, not placeholder letters such as x, y, or v.\n"
    "If no clear visible real person exists, output exactly:\n"
    "<answer></answer>\n\n"
    "Begin the answer now. Output tags only.\n"
    "Answer:"
)


PROMPT_BOX_POSE_ATTR_COMPACT_TAG = (
    "<image>\n"
    "You are a human-structure annotation model.\n\n"
    "Return exactly one compact XML block and stop after </answer>.\n"
    "Allowed tags only: <answer>, </answer>, <p>, </p>, <b>, </b>, <k>, </k>, <g>, </g>, <a>, </a>, <s>, </s>\n"
    "No prose, markdown, JSON, bullets, comments, XML attributes, schema explanation, or chain-of-thought.\n\n"
    "Detect every clear visible real person exactly once. Include clear partially visible people. Skip reflections, posters, mannequins, body parts without a clear person instance, and ambiguous tiny regions. Do not duplicate the same person. Output persons in left-to-right order by box center, breaking ties top-to-bottom. Do not use a fixed maximum person count.\n\n"
    "Coordinates are normalized integers in [0,1000]; x is left-to-right and y is top-to-bottom.\n\n"
    "For each person output this numeric tag structure:\n"
    "<p><b>120,250,480,820</b><k>250,300,2;260,280,2;240,280,2;280,290,2;220,290,2;300,350,2;200,350,2;350,450,2;150,450,2;380,500,1;120,500,2;280,600,2;220,600,2;290,750,1;210,750,1;0,0,0;0,0,0</k><g>0</g><a>25</a><s>1</s></p>\n\n"
    "<b> is x1,y1,x2,y2 with 0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000.\n"
    "<k> has exactly 17 keypoint triplets separated by semicolons, exactly 51 numbers, and exactly 16 semicolons.\n"
    "The COCO-17 order is: "
    f"{COCO_KEYPOINT_ORDER}.\n"
    "Each keypoint triplet is horizontal coordinate, vertical coordinate, visibility. Visibility is 2 for visible, 1 for occluded but inferable, and 0 for missing/outside/not reliable. If visibility is 0, output exactly 0,0,0.\n"
    "Numeric fields must contain digits only, not placeholder letters such as x, y, or v.\n"
    "<g> is 0 male, 1 female, -1 unknown.\n"
    "<a> is estimated age in years, or -1 unknown.\n"
    "<s> is confidence in [0,1]. If scores are not calibrated during training, use 1.\n\n"
    "If no clear visible real person exists, output exactly <answer></answer>.\n"
    "Output tags only.\n"
    "Answer:"
)


def build_box_pose_attribute_prompt(
    extra_instruction: Optional[str] = None,
    output_format: str = FORMAT_TAG,
) -> str:
    fmt = normalize_target_format(output_format)
    base = PROMPT_BOX_POSE_ATTR_TAG if fmt == FORMAT_TAG else PROMPT_BOX_POSE_ATTR_COMPACT_TAG
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base


def build_vl_chat_text(processor, prompt: str, assistant_prefix: str = "") -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<image>\n{prompt}\nAnswer:"
    if assistant_prefix:
        text += assistant_prefix
    return text


def strip_answer_prefix(text: str) -> str:
    if not isinstance(text, str):
        return ""
    stripped = text.lstrip()
    if stripped.startswith("<answer>"):
        stripped = stripped[len("<answer>") :]
        stripped = stripped.lstrip("\r\n")
    return stripped
