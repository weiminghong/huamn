from typing import Optional


COCO_KEYPOINT_ORDER = (
    "nose,left_eye,right_eye,left_ear,right_ear,"
    "left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,"
    "left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle"
)


def build_box_pose_attribute_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "Detect all persons in the image and output one valid JSON object only.\n"
        "Required JSON schema:\n"
        '{\n'
        '  "persons": [\n'
        '    {\n'
        '      "person_id": 1,\n'
        '      "bbox_2d": [x1, y1, x2, y2],\n'
        '      "bbox_format": "norm_xyxy_1000",\n'
        '      "keypoints_2d": [[x, y, v], ... 17 items ...],\n'
        '      "keypoint_format": "norm_xyv_1000",\n'
        '      "gender": 0,\n'
        '      "age": 25,\n'
        '      "score": 1.0\n'
        '    }\n'
        '  ]\n'
        '}\n'
        "Rules:\n"
        "1) Output JSON only. Do not output markdown, XML, analysis, or explanations.\n"
        "2) The only top-level field is persons. Use an empty list when no person is visible.\n"
        "3) bbox_2d is [x1,y1,x2,y2] normalized to a 0-1000 coordinate grid.\n"
        "4) keypoints_2d contains exactly 17 COCO keypoints in this order: "
        f"{COCO_KEYPOINT_ORDER}.\n"
        "5) Each keypoint is [x,y,v], where x,y are normalized to 0-1000 and v is 0 missing, 1 occluded/inferred, 2 visible.\n"
        "6) If a keypoint is missing, output exactly [0,0,0].\n"
        "7) gender uses COCO-UniHuman coding: 0 male, 1 female, -1 unknown.\n"
        "8) age is estimated age in years; use -1 when unknown.\n"
        "9) score is confidence in [0,1]. Use 1.0 for ground-truth style answers.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
