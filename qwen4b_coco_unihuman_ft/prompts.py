from typing import Optional


COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)


PROMPT_BOX_POSE_ATTR_TAG = (
    "<image>\n"
    "Detect every visible person in the image.\n"
    'Output only XML blocks wrapped inside <answer>...</answer>. No explanation.\n\n'
    "Each <person> block must contain <box>, <pose>, <gender>, <age>, and <score> tags.\n\n"
    "Rules:\n"
    "- Coordinates are normalized to [0,1000].\n"
    "- Box uses [x1,y1,x2,y2].\n"
    "- Pose contains exactly 17 keypoints in COCO order:\n"
    "  "
    f"{COCO_KEYPOINT_ORDER}.\n"
    "- Format: Each keypoint is a triplet [x, y, v].\n"
    "  * x, y: coordinates.\n"
    "  * v (visibility): 2 = visible, 1 = occluded/inferred, 0 = missing/out-of-frame.\n"
    "- In <pose>, output exactly 17 comma-separated [x, y, v] triplets, no more and no fewer.\n"
    "- If uncertain, still output 17 triplets and use [0,0,0] for unknown keypoints.\n"
    "- If a keypoint is missing (v=0), you MUST output exactly [0,0,0].\n"
    "- Gender: 0 male, 1 female, -1 unknown.\n"
    "- Age is estimated age in years; use -1 when unknown.\n"
    # "- Score is confidence in [0,1]. Use 1.0 for ground-truth style answers.\n"
    "- Output tags only. Do not output markdown, JSON, analysis, or explanations.\n"
    "\n--- EXAMPLE OUTPUT ---\n"
    "<answer>\n"
    "<person>\n"
    "<box>[120, 250, 480, 820]</box>\n"
    "<pose>[250, 300, 2], [260, 280, 2], [240, 280, 2], [280, 290, 2], [220, 290, 2], [300, 350, 2], [200, 350, 2], [350, 450, 2], [150, 450, 2], [380, 500, 1], [120, 500, 2], [280, 600, 2], [220, 600, 2], [290, 750, 1], [210, 750, 1], [0, 0, 0], [0, 0, 0]</pose>\n"
    "<gender>0</gender>\n"
    "<age>25</age>\n"
    "<score>0.98</score>\n"
    "</person>\n"
    "</answer>\n"
    "--- END OF EXAMPLE ---\n"
    "Answer:"
)


def build_box_pose_attribute_prompt(extra_instruction: Optional[str] = None) -> str:
    base = PROMPT_BOX_POSE_ATTR_TAG
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
