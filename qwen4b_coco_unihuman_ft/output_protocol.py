import re
from typing import Any, Dict, List, Optional, Tuple


FORMAT_TAG = "tag"
FORMAT_COMPACT_TAG = "compact_tag"
SUPPORTED_TARGET_FORMATS = (FORMAT_TAG, FORMAT_COMPACT_TAG)


def normalize_target_format(value: Any) -> str:
    fmt = str(value or FORMAT_COMPACT_TAG).strip().lower().replace("-", "_")
    aliases = {
        "compact": FORMAT_COMPACT_TAG,
        "short": FORMAT_COMPACT_TAG,
        "short_tag": FORMAT_COMPACT_TAG,
        "xml": FORMAT_TAG,
        "long_tag": FORMAT_TAG,
    }
    fmt = aliases.get(fmt, fmt)
    if fmt not in SUPPORTED_TARGET_FORMATS:
        raise ValueError(f"Unsupported target/output format: {value!r}. Use one of {SUPPORTED_TARGET_FORMATS}.")
    return fmt


def _to_float(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out or out in (float("inf"), float("-inf")):
        return None
    return out


def _norm_x(x: Any, width: int) -> int:
    if width <= 0:
        return 0
    return int(round(max(0.0, min(float(x), float(width - 1))) / float(width) * 1000.0))


def _norm_y(y: Any, height: int) -> int:
    if height <= 0:
        return 0
    return int(round(max(0.0, min(float(y), float(height - 1))) / float(height) * 1000.0))


def _denorm_x(x_norm: Any, width: int) -> float:
    if width <= 0:
        return 0.0
    return max(0.0, min(float(x_norm) / 1000.0 * float(width), float(max(0, width - 1))))


def _denorm_y(y_norm: Any, height: int) -> float:
    if height <= 0:
        return 0.0
    return max(0.0, min(float(y_norm) / 1000.0 * float(height), float(max(0, height - 1))))


def _attrs_by_person_id(attrs: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item.get("person_id"): item for item in attrs if isinstance(item, dict)}


def _poses_by_person_id(poses: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item.get("person_id"): item for item in poses if isinstance(item, dict)}


def norm_pose_17(pose: Optional[Dict[str, Any]], width: int, height: int) -> List[List[int]]:
    raw = pose.get("keypoints", []) if isinstance(pose, dict) else []
    out: List[List[int]] = []
    keypoints = raw[:17] if isinstance(raw, list) else []
    for kp in keypoints:
        if not isinstance(kp, list) or len(kp) < 3:
            out.append([0, 0, 0])
            continue
        v = int(round(float(kp[2])))
        if v <= 0:
            out.append([0, 0, 0])
        else:
            out.append([_norm_x(kp[0], width), _norm_y(kp[1], height), 2 if v > 1 else 1])
    while len(out) < 17:
        out.append([0, 0, 0])
    return out[:17]


def _format_long_pose(keypoints: List[List[int]]) -> str:
    return ", ".join(f"[{int(kp[0])}, {int(kp[1])}, {int(kp[2])}]" for kp in keypoints[:17])


def _format_compact_pose(keypoints: List[List[int]]) -> str:
    return ";".join(f"{int(kp[0])},{int(kp[1])},{int(kp[2])}" for kp in keypoints[:17])


def build_target_text(record: Dict[str, Any], target_format: str = FORMAT_COMPACT_TAG) -> str:
    fmt = normalize_target_format(target_format)
    width = int(record.get("width", 0))
    height = int(record.get("height", 0))
    poses_by_pid = _poses_by_person_id(record.get("pose", []))
    attrs_by_pid = _attrs_by_person_id(record.get("attribute", []))
    person_blocks: List[str] = []
    for box_item in record.get("box", []):
        if not isinstance(box_item, dict):
            continue
        bbox = box_item.get("bbox_xyxy", [])
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        pid = box_item.get("person_id", len(person_blocks) + 1)
        attr = attrs_by_pid.get(pid, {})
        box = [
            _norm_x(bbox[0], width),
            _norm_y(bbox[1], height),
            _norm_x(bbox[2], width),
            _norm_y(bbox[3], height),
        ]
        pose = norm_pose_17(poses_by_pid.get(pid), width, height)
        gender = int(attr.get("gender", -1)) if isinstance(attr, dict) else -1
        age = int(round(float(attr.get("age", -1)))) if isinstance(attr, dict) and "age" in attr else -1
        if fmt == FORMAT_COMPACT_TAG:
            person_blocks.append(
                f"<p><b>{box[0]},{box[1]},{box[2]},{box[3]}</b>"
                f"<k>{_format_compact_pose(pose)}</k>"
                f"<g>{gender}</g><a>{age}</a><s>1</s></p>"
            )
        else:
            person_blocks.append(
                "\n".join(
                    [
                        "<person>",
                        f"<box>[{box[0]}, {box[1]}, {box[2]}, {box[3]}]</box>",
                        f"<pose>{_format_long_pose(pose)}</pose>",
                        f"<gender>{gender}</gender>",
                        f"<age>{age}</age>",
                        "<score>1.0</score>",
                        "</person>",
                    ]
                )
            )
    if not person_blocks:
        return "<answer></answer>"
    if fmt == FORMAT_COMPACT_TAG:
        return "<answer>" + "".join(person_blocks) + "</answer>"
    return "<answer>\n" + "\n".join(person_blocks) + "\n</answer>"


def _answer_body(text: str, require_complete: bool = False) -> Optional[str]:
    if not isinstance(text, str) or not text:
        return None
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1)
    if require_complete:
        return None
    start = re.search(r"<answer>", text, flags=re.IGNORECASE)
    if start:
        return text[start.end():]
    return text


def parse_compact_tag_struct(
    text: str,
    image_size: Optional[Tuple[int, int]] = None,
    require_complete_answer: bool = False,
) -> Optional[Dict[str, Any]]:
    if image_size is None:
        return None
    body = _answer_body(text, require_complete=require_complete_answer)
    if body is None:
        return None
    if not body.strip():
        return {"box": [], "pose": [], "attribute": []}

    width, height = image_size
    num_pattern = r"-?(?:\d+\.\d+|\d+|\.\d+)"
    boxes: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    attrs: List[Dict[str, Any]] = []
    for person_id, m_person in enumerate(re.finditer(r"<p>(.*?)</p>", body, flags=re.DOTALL | re.IGNORECASE), start=1):
        blob = m_person.group(1)
        m_box = re.search(r"<b>(.*?)</b>", blob, flags=re.DOTALL | re.IGNORECASE)
        m_pose = re.search(r"<k>(.*?)</k>", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_box is None or m_pose is None:
            continue
        box_nums = re.findall(num_pattern, m_box.group(1))
        pose_nums = re.findall(num_pattern, m_pose.group(1))
        if len(box_nums) != 4 or len(pose_nums) != 51:
            continue
        x1n, y1n, x2n, y2n = [float(x) for x in box_nums[:4]]
        x1 = _denorm_x(x1n, width)
        y1 = _denorm_y(y1n, height)
        x2 = _denorm_x(x2n, width)
        y2 = _denorm_y(y2n, height)
        if x2 <= x1 or y2 <= y1:
            continue
        score = 1.0
        m_score = re.search(r"<s>(.*?)</s>", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_score is not None:
            score_val = _to_float(m_score.group(1).strip())
            if score_val is not None:
                score = max(0.0, min(float(score_val), 1.0))
        boxes.append({"person_id": person_id, "bbox_xyxy": [x1, y1, x2, y2], "score": score})

        kpts: List[List[float]] = []
        for i in range(17):
            x_n = float(pose_nums[i * 3 + 0])
            y_n = float(pose_nums[i * 3 + 1])
            v_n = int(round(float(pose_nums[i * 3 + 2])))
            if x_n == 0 and y_n == 0 and v_n == 0:
                kpts.append([0.0, 0.0, 0.0])
            else:
                if v_n not in (0, 1, 2):
                    v_n = 0
                kpts.append([_denorm_x(x_n, width), _denorm_y(y_n, height), float(v_n)])
        poses.append({"person_id": person_id, "keypoints": kpts, "score": score})

        attr_item: Dict[str, Any] = {"person_id": person_id}
        m_gender = re.search(r"<g>(.*?)</g>", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_gender is not None:
            gender = _to_float(m_gender.group(1).strip())
            if gender is not None and int(round(gender)) in (0, 1):
                attr_item["gender"] = int(round(gender))
        m_age = re.search(r"<a>(.*?)</a>", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_age is not None:
            age = _to_float(m_age.group(1).strip())
            if age is not None and age >= 0:
                attr_item["age"] = float(age)
        if len(attr_item) > 1:
            attrs.append(attr_item)
    if not boxes:
        return None
    return {"box": boxes, "pose": poses, "attribute": attrs}


def _text_with_matches_removed(text: str, matches: List[re.Match]) -> str:
    pieces: List[str] = []
    last = 0
    for match in matches:
        pieces.append(text[last : match.start()])
        last = match.end()
    pieces.append(text[last:])
    return "".join(pieces)


def audit_tag_format(text: str, output_format: str = FORMAT_TAG) -> Dict[str, Any]:
    fmt = normalize_target_format(output_format)
    raw = text if isinstance(text, str) else ""
    issues: List[str] = []
    allowed_tags = (
        {"answer", "person", "box", "pose", "gender", "age", "score"}
        if fmt == FORMAT_TAG
        else {"answer", "p", "b", "k", "g", "a", "s"}
    )
    person_tag = "person" if fmt == FORMAT_TAG else "p"
    fields = ("box", "pose", "gender", "age", "score") if fmt == FORMAT_TAG else ("b", "k", "g", "a", "s")
    pose_tag = "pose" if fmt == FORMAT_TAG else "k"

    answer_match = re.fullmatch(r"\s*<answer>(.*?)</answer>\s*", raw, flags=re.DOTALL | re.IGNORECASE)
    if answer_match is None:
        issues.append("answer_wrapper_or_extra")
        body_match = re.search(r"<answer>(.*?)</answer>", raw, flags=re.DOTALL | re.IGNORECASE)
        body = body_match.group(1) if body_match else raw
    else:
        body = answer_match.group(1)

    for m_tag in re.finditer(r"<(/?)([A-Za-z_][\w:-]*)([^>]*)>", raw):
        tag = m_tag.group(2).lower()
        attrs = m_tag.group(3).strip()
        if tag not in allowed_tags:
            issues.append(f"bad_tag:{tag}")
        if attrs:
            issues.append(f"tag_attributes:{tag}")

    open_re = fr"<{person_tag}>"
    close_re = fr"</{person_tag}>"
    open_count = len(re.findall(open_re, body, flags=re.IGNORECASE))
    close_count = len(re.findall(close_re, body, flags=re.IGNORECASE))
    if open_count != close_count:
        issues.append(f"person_unclosed:{open_count}!={close_count}")

    block_re = fr"<{person_tag}>(.*?)</{person_tag}>"
    person_matches = list(re.finditer(block_re, body, flags=re.DOTALL | re.IGNORECASE))
    if open_count and len(person_matches) != open_count:
        issues.append(f"person_block_parse:{len(person_matches)}/{open_count}")
    residual = _text_with_matches_removed(body, person_matches).strip()
    if residual and open_count == close_count:
        issues.append("body_stray_content")

    num_pattern = r"-?(?:\d+\.\d+|\d+|\.\d+)"
    for person_idx, match in enumerate(person_matches, start=1):
        block = match.group(1)
        for field in fields:
            field_matches = re.findall(fr"<{field}>(.*?)</{field}>", block, flags=re.DOTALL | re.IGNORECASE)
            if len(field_matches) != 1:
                issues.append(f"person{person_idx}_{field}_tag")
        pose_match = re.search(fr"<{pose_tag}>(.*?)</{pose_tag}>", block, flags=re.DOTALL | re.IGNORECASE)
        if pose_match is not None:
            pose_text = pose_match.group(1)
            if re.search(r"[A-Za-z]", pose_text):
                issues.append(f"person{person_idx}_pose_letters")
            pose_nums = re.findall(num_pattern, pose_text)
            if len(pose_nums) != 51:
                issues.append(f"person{person_idx}_pose_nums:{len(pose_nums)}")
            elif any(int(round(float(v))) not in (0, 1, 2) for v in pose_nums[2::3]):
                issues.append(f"person{person_idx}_bad_visibility")

    # Deduplicate while preserving order so row JSON stays compact and stable.
    unique_issues = list(dict.fromkeys(issues))
    return {
        "ok": not unique_issues,
        "issues": unique_issues,
        "open_person_blocks": int(open_count),
        "closed_person_blocks": int(close_count),
        "parsed_person_blocks": int(len(person_matches)),
    }
