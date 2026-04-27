from typing import Optional


def build_box_pose_attribute_prompt(extra_instruction: Optional[str] = None) -> str:
    base = (
        "<image>\n"
        "Detect all persons and output one JSON object only.\n"
        "Required top-level fields:\n"
        '- "box": list of person boxes, each item uses "person_id" and "bbox_xyxy".\n'
        '- "pose": list of person keypoints, each item uses "person_id" and "keypoints".\n'
        '- "attribute": list of person attributes, each item uses "person_id" and may include "gender" and "age".\n'
        "Rules:\n"
        "1) Always output all three fields; use [] when missing.\n"
        "2) Use pixel coordinates from the original image.\n"
        "3) bbox_xyxy means [x1,y1,x2,y2].\n"
        "4) keypoints use COCO-17 order: nose,left_eye,right_eye,left_ear,right_ear,left_shoulder,right_shoulder,left_elbow,right_elbow,left_wrist,right_wrist,left_hip,right_hip,left_knee,right_knee,left_ankle,right_ankle.\n"
        "5) Each keypoint is [x,y,v], where v is 0 missing, 1 labeled but not visible, 2 visible.\n"
        "6) Use the same person_id to align box, pose, and attribute entries.\n"
        "7) Return valid JSON only. No explanation.\n"
        "Answer:"
    )
    if extra_instruction:
        return f"{base}\n{extra_instruction}"
    return base
