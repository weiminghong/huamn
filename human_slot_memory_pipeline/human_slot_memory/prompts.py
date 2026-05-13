COCO17_KEYPOINT_ORDER = (
    "nose, left_eye, right_eye, left_ear, right_ear, "
    "left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, "
    "left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle"
)

DEFAULT_ASSISTANT_PREFIX_STAGE0 = "<answer>\n<person>\n<box>["
DEFAULT_ASSISTANT_PREFIX_STAGE1 = "<answer>\n"


STRUCTURED_HUMAN_STAGE0_PROMPT = f"""Detect every visible person in the image.
Output only XML blocks wrapped inside <answer>...</answer>. No explanation.

Each <person> block must contain <box>, <pose>, <gender>, <age>, and <score> tags.

Rules:
- Output exactly one <answer>...</answer> block.
- Do not output <think>, </think>, markdown, JSON, analysis, or duplicated answers.
- Coordinates are normalized to [0,1000].
- Box uses [x1,y1,x2,y2].
- Pose contains exactly 17 keypoints in COCO order:
  {COCO17_KEYPOINT_ORDER}.
- Each keypoint must be a triplet [x, y, v].
- v: 2 visible, 1 occluded/inferred, 0 missing.
- In <pose>, output exactly 17 comma-separated [x, y, v] triplets.
- If a keypoint is missing, output exactly [0,0,0].
- Gender: 0 male, 1 female, -1 unknown.
- Age: use years if known, otherwise -1.
- If no visible person exists, output exactly <answer></answer>.

Example:
<answer>
<person>
<box>[120,250,480,820]</box>
<pose>[250,300,2], [260,280,2], [240,280,2], [280,290,2], [220,290,2], [300,350,2], [200,350,2], [350,450,2], [150,450,2], [380,500,1], [120,500,2], [280,600,2], [220,600,2], [290,750,1], [210,750,1], [0,0,0], [0,0,0]</pose>
<gender>0</gender>
<age>25</age>
<score>0.98</score>
</person>
</answer>"""


STAGE0_REPAIR_PROMPT = f"""Rewrite the raw output into strict XML only.
Output exactly one <answer>...</answer> block and nothing else.

Rules:
- Remove all thinking, markdown, JSON, and duplicate answer blocks.
- Each <person> block must contain <box>, <pose>, <gender>, <age>, and <score>.
- Box coordinates are normalized to [0,1000] using [x1,y1,x2,y2].
- Pose must contain exactly 17 [x, y, v] triplets in COCO order:
  {COCO17_KEYPOINT_ORDER}.
- If a keypoint is missing, output [0,0,0].
- Gender: 0 male, 1 female, -1 unknown.
- Age: use years if known, otherwise -1.
- If no visible person exists, output exactly <answer></answer>.

RAW OUTPUT:
"""


STRUCTURED_HUMAN_PROBE_PROMPT = """<image>
Detect every visible person in the image.
Output only XML blocks wrapped inside <answer>...</answer>. No explanation.

Each <person> block must contain <box>, <pose>, <gender>, <age>, and <score> tags.

Rules:
- Coordinates are normalized to [0,1000].
- Box uses [x1,y1,x2,y2].
- Pose contains exactly 17 keypoints in COCO order.
- Each keypoint is [x, y, v].
- v: 2 visible, 1 occluded/inferred, 0 missing.
- If a keypoint is missing, output exactly [0,0,0].
- Gender: 0 male, 1 female, -1 unknown.
- Age: use years if known, otherwise -1.
- Output tags only. Do not output markdown, JSON, analysis, or thinking.

Example:
<answer>
<person>
<box>[120,250,480,820]</box>
<pose>[250,300,2], [260,280,2], [240,280,2], [280,290,2], [220,290,2], [300,350,2], [200,350,2], [350,450,2], [150,450,2], [380,500,1], [120,500,2], [280,600,2], [220,600,2], [290,750,1], [210,750,1], [0,0,0], [0,0,0]</pose>
<gender>0</gender>
<age>25</age>
<score>0.98</score>
</person>
</answer>
Answer:"""


STRUCTURED_HUMAN_PROMPT = """<image>
Detect every visible person and describe them as structured human slots.
Output only XML blocks wrapped inside <answer>...</answer>. No explanation.

Each <person> block may contain:
- <box>[x1,y1,x2,y2]</box> using absolute image coordinates.
- <pose> with exactly 17 keypoints in COCO order.
- <parsing>, <attr>, <action>, <relation>, <caption>.

Rules:
- Output tags only. Do not output markdown, JSON, analysis, or thinking.
- In <pose>, output exactly 17 [x, y, v] triplets.
- v uses: 2 visible, 1 occluded/inferred, 0 missing.
- If a keypoint is missing, output [0,0,0].
- Use "unknown" only for tasks that are truly unavailable.

Example:
<answer>
<person id="0">
<box>[120,250,480,820]</box>
<pose>[250,300,2], [260,280,2], [240,280,2], [280,290,2], [220,290,2], [300,350,2], [200,350,2], [350,450,2], [150,450,2], [380,500,1], [120,500,2], [280,600,2], [220,600,2], [290,750,1], [210,750,1], [0,0,0], [0,0,0]</pose>
<parsing>unknown</parsing>
<attr><gender>unknown</gender><age>unknown</age></attr>
<action>unknown</action>
<relation>unknown</relation>
<caption>person standing</caption>
</person>
</answer>
Answer:"""


def build_stage1_prompt(extra_prompt: str | None = None) -> str:
    if extra_prompt:
        return STRUCTURED_HUMAN_PROMPT + "\n" + extra_prompt.strip()
    return STRUCTURED_HUMAN_PROMPT


def build_stage0_probe_prompt() -> str:
    return STRUCTURED_HUMAN_STAGE0_PROMPT


def build_stage1_probe_prompt() -> str:
    return STRUCTURED_HUMAN_PROBE_PROMPT


def build_stage3_prompt(question: str | None = None) -> str:
    if question:
        return "Answer using the provided Human Memory Tokens.\nQuestion: " + question.strip()
    return "Generate a grounded human-centric caption using the provided Human Memory Tokens."
