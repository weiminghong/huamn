from typing import Optional


COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)


PROMPT_BOX_POSE_ATTR_TAG = (
    "<image>\n"
    "Detect every visible person in the image, including small, occluded, and edge persons.\n"
    "Output only XML-like tags wrapped inside <answer>...</answer>.\n\n"
    "Required output format:\n"
    "<answer>\n"
    "<person>\n"
    "<box>[x1, y1, x2, y2]</box>\n"
    "<pose>[x, y, v], ... exactly 17 keypoints ...</pose>\n"
    "<gender>0</gender>\n"
    "<age>25</age>\n"
    "<score>1.0</score>\n"
    "</person>\n"
    "</answer>\n\n"
    "Rules:\n"
    "- Coordinates are normalized to [0,1000].\n"
    "- Box uses [x1,y1,x2,y2].\n"
    "- Pose contains exactly 17 COCO keypoints in this order: "
    f"{COCO_KEYPOINT_ORDER}.\n"
    "- Each keypoint is [x,y,v], where v=2 visible, v=1 occluded/inferred, v=0 missing/out-of-frame.\n"
    "- If a keypoint is missing, output exactly [0,0,0].\n"
    "- Gender uses COCO-UniHuman coding: 0 male, 1 female, -1 unknown.\n"
    "- Age is estimated age in years; use -1 when unknown.\n"
    "- Score is confidence in [0,1]. Use 1.0 for ground-truth style answers.\n"
    "- If no person is visible, output exactly <answer></answer>.\n"
    "- Output tags only. Do not output markdown, JSON, analysis, or explanations.\n"
    "Answer:"
)


def build_box_pose_attribute_prompt(extra_instruction: Optional[str] = None) -> str:
    base = PROMPT_BOX_POSE_ATTR_TAG
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
