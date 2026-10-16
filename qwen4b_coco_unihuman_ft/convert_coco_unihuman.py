import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from qwen4b_coco_unihuman_ft.prompts import build_box_pose_attribute_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO-UniHuman annotations to box/pose/attribute SFT JSONL."
    )
    parser.add_argument("--train-ann-json", type=str, default="", help="Train annotation json.")
    parser.add_argument("--val-ann-json", type=str, default="", help="Val annotation json.")
    parser.add_argument("--ann-json", type=str, default="", help="Single annotation json (split by ratio).")
    parser.add_argument("--image-root", type=str, required=True, help="Default image root directory.")
    parser.add_argument("--train-image-root", type=str, default="", help="Optional train image root directory.")
    parser.add_argument("--val-image-root", type=str, default="", help="Optional val image root directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for train/val jsonl.")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--skip-missing-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip records whose local image file does not exist.",
    )
    parser.add_argument(
        "--emit-swift-format",
        action="store_true",
        help="Also emit ms-swift format jsonl: train_swift.jsonl / val_swift.jsonl.",
    )
    return parser.parse_args()


def _xywh_to_xyxy(bbox_xywh: List[Any]) -> List[float]:
    x, y, w, h = [float(v) for v in bbox_xywh]
    return [x, y, x + w, y + h]


def _flat_kpts_to_triplets(flat_kpts: List[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    n = len(flat_kpts) // 3
    for i in range(n):
        x = float(flat_kpts[3 * i + 0])
        y = float(flat_kpts[3 * i + 1])
        v = float(flat_kpts[3 * i + 2])
        out.append([x, y, v])
    return out


def _to_rel_image_path(image_root: str, file_name: str) -> str:
    file_name = str(file_name).replace("\\", "/")
    p = Path(file_name)
    if p.is_absolute():
        try:
            return str(p.relative_to(Path(image_root))).replace("\\", "/")
        except Exception:
            return str(p).replace("\\", "/")
    return file_name


def _dump_jsonl(path: str, records: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _norm_x(x: Any, width: int) -> int:
    if width <= 0:
        return 0
    return int(round(max(0.0, min(float(x), float(width - 1))) / float(width) * 1000.0))


def _norm_y(y: Any, height: int) -> int:
    if height <= 0:
        return 0
    return int(round(max(0.0, min(float(y), float(height - 1))) / float(height) * 1000.0))


def _attrs_by_person_id(attrs: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item.get("person_id"): item for item in attrs if isinstance(item, dict)}


def _poses_by_person_id(poses: List[Dict[str, Any]]) -> Dict[Any, Dict[str, Any]]:
    return {item.get("person_id"): item for item in poses if isinstance(item, dict)}


def _norm_pose_17(pose: Optional[Dict[str, Any]], width: int, height: int) -> List[List[int]]:
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


def _format_tag_pose(keypoints: List[List[int]]) -> str:
    return ", ".join(f"[{int(kp[0])}, {int(kp[1])}, {int(kp[2])}]" for kp in keypoints[:17])


def _build_target_text(record: Dict[str, Any]) -> str:
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
        pose = _norm_pose_17(poses_by_pid.get(pid), width, height)
        gender = int(attr.get("gender", -1)) if isinstance(attr, dict) else -1
        age = int(round(float(attr.get("age", -1)))) if isinstance(attr, dict) and "age" in attr else -1
        person_blocks.append(
            "\n".join(
                [
                    "<person>",
                    f"<box>[{box[0]}, {box[1]}, {box[2]}, {box[3]}]</box>",
                    f"<pose>{_format_tag_pose(pose)}</pose>",
                    f"<gender>{gender}</gender>",
                    f"<age>{age}</age>",
                    "<score>1.0</score>",
                    "</person>",
                ]
            )
        )
    if not person_blocks:
        return "<answer></answer>"
    return "<answer>\n" + "\n".join(person_blocks) + "\n</answer>"


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid json root, expected dict: {path}")
    return data


def _is_valid_ann_for_field(ann: Dict[str, Any], field: str, fallback_default: bool) -> bool:
    valid = ann.get("valid", None)
    if isinstance(valid, dict):
        v = valid.get(field, 0)
        try:
            return bool(int(v))
        except Exception:
            return bool(v)
    return fallback_default


def _to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _to_int_if_possible(value: Any) -> Any:
    num = _to_number(value)
    if num is None:
        return value
    if float(num).is_integer():
        return int(num)
    return num


def _extract_from_instances(
    instances: Iterable[Dict[str, Any]],
    max_keypoints: int = 17,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    boxes: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []
    attrs: List[Dict[str, Any]] = []
    stats = {
        "num_instances": 0,
        "num_boxes": 0,
        "num_poses": 0,
        "num_attributes": 0,
        "num_gender": 0,
        "num_age": 0,
    }
    person_id = 0
    for ann in instances:
        if not isinstance(ann, dict):
            continue
        stats["num_instances"] += 1

        has_box = _is_valid_ann_for_field(ann, "bboxes", fallback_default=True)
        bbox = ann.get("bbox", [])
        if not (has_box and isinstance(bbox, list) and len(bbox) == 4):
            continue
        x1, y1, x2, y2 = _xywh_to_xyxy(bbox)
        if x2 <= x1 or y2 <= y1:
            continue

        person_id += 1
        boxes.append({"person_id": person_id, "bbox_xyxy": [x1, y1, x2, y2]})
        stats["num_boxes"] += 1

        has_kpt = _is_valid_ann_for_field(ann, "keypoints_2d", fallback_default=True)
        kpts = ann.get("keypoints", [])
        if has_kpt and isinstance(kpts, list) and len(kpts) >= 3:
            triplets = _flat_kpts_to_triplets(kpts)[:max_keypoints]
            if len(triplets) > 0:
                poses.append({"person_id": person_id, "keypoints": triplets})
                stats["num_poses"] += 1

        attr_item: Dict[str, Any] = {}
        if _is_valid_ann_for_field(ann, "gender", fallback_default=("gender" in ann)) and "gender" in ann:
            attr_item["gender"] = _to_int_if_possible(ann["gender"])
            stats["num_gender"] += 1
        if _is_valid_ann_for_field(ann, "age", fallback_default=("age" in ann)) and "age" in ann:
            attr_item["age"] = _to_int_if_possible(ann["age"])
            stats["num_age"] += 1
        if attr_item:
            attr_item = {"person_id": person_id, **attr_item}
            attrs.append(attr_item)
            stats["num_attributes"] += 1
    return boxes, poses, attrs, stats


def _convert_payload(payload: Dict[str, Any], image_root: str, skip_missing_images: bool) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    images = payload.get("images", [])
    if not isinstance(images, list):
        raise ValueError("Invalid format: images must be list.")

    annotations = payload.get("annotations", None)
    anns_by_img: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    use_coco_annotations = isinstance(annotations, list)
    if use_coco_annotations:
        for ann in annotations:
            if not isinstance(ann, dict):
                continue
            image_id = ann.get("image_id", None)
            if image_id is None:
                continue
            try:
                anns_by_img[int(image_id)].append(ann)
            except Exception:
                continue

    converted: List[Dict[str, Any]] = []
    stats = {
        "num_images_seen": 0,
        "num_images_kept": 0,
        "num_images_missing": 0,
        "num_instances": 0,
        "num_boxes": 0,
        "num_poses": 0,
        "num_attributes": 0,
        "num_gender": 0,
        "num_age": 0,
    }
    for i, img in enumerate(images):
        stats["num_images_seen"] += 1
        image_id = int(img.get("id", i))
        file_name = str(img.get("file_name", "")).strip()
        if not file_name:
            continue

        image_rel = _to_rel_image_path(image_root, file_name)
        image_abs = Path(image_root) / image_rel
        if not image_abs.exists():
            stats["num_images_missing"] += 1
            if skip_missing_images:
                continue

        if use_coco_annotations:
            img_anns = anns_by_img.get(image_id, [])
            img_anns = sorted(img_anns, key=lambda x: int(x.get("id", 0)))
        else:
            img_anns = img.get("instance", [])
            if not isinstance(img_anns, list):
                img_anns = []

        boxes, poses, attrs, img_stats = _extract_from_instances(img_anns)
        for key, value in img_stats.items():
            stats[key] += value
        rec = {
            "id": str(image_id),
            "image": image_rel,
            "image_abs": str(image_abs),
            "width": int(img.get("width", 0)),
            "height": int(img.get("height", 0)),
            "box": boxes,
            "pose": poses,
            "attribute": attrs,
        }
        rec["target_text"] = _build_target_text(rec)
        converted.append(rec)
        stats["num_images_kept"] += 1
    return converted, stats


def _to_swift_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    swift_records: List[Dict[str, Any]] = []
    prompt = build_box_pose_attribute_prompt()
    for rec in records:
        target_text = rec.get("target_text", _build_target_text(rec))
        image_path = rec.get("image_abs", rec.get("image", ""))
        swift_records.append(
            {
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": target_text},
                ],
                "images": [image_path],
            }
        )
    return swift_records


def _apply_max_samples(records: List[Dict[str, Any]], max_samples: Optional[int]) -> List[Dict[str, Any]]:
    if max_samples is not None:
        return records[: int(max_samples)]
    return records


def _save_split(output_dir: Path, split: str, records: List[Dict[str, Any]], emit_swift: bool) -> Dict[str, str]:
    out: Dict[str, str] = {}
    jsonl_path = output_dir / f"{split}.jsonl"
    _dump_jsonl(str(jsonl_path), records)
    out[f"output_{split}_jsonl"] = str(jsonl_path)

    if emit_swift:
        swift_path = output_dir / f"{split}_swift.jsonl"
        _dump_jsonl(str(swift_path), _to_swift_records(records))
        out[f"output_{split}_swift_jsonl"] = str(swift_path)
    return out


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.train_ann_json and args.val_ann_json:
        train_image_root = args.train_image_root or args.image_root
        val_image_root = args.val_image_root or args.image_root
        train_payload = _read_json(args.train_ann_json)
        val_payload = _read_json(args.val_ann_json)
        train_records_all, train_stats = _convert_payload(
            train_payload, train_image_root, skip_missing_images=bool(args.skip_missing_images)
        )
        val_records_all, val_stats = _convert_payload(
            val_payload, val_image_root, skip_missing_images=bool(args.skip_missing_images)
        )
        train_records = _apply_max_samples(train_records_all, args.max_samples)
        val_records = _apply_max_samples(val_records_all, args.max_samples)

        outputs = {}
        outputs.update(_save_split(output_dir, "train", train_records, args.emit_swift_format))
        outputs.update(_save_split(output_dir, "val", val_records, args.emit_swift_format))
        summary = {
            "mode": "separate_train_val_json",
            "train_ann_json": args.train_ann_json,
            "val_ann_json": args.val_ann_json,
            "image_root": args.image_root,
            "train_image_root": train_image_root,
            "val_image_root": val_image_root,
            "num_train": len(train_records),
            "num_val": len(val_records),
            "train_stats": train_stats,
            "val_stats": val_stats,
            **outputs,
        }
    elif args.ann_json:
        if args.train_ratio <= 0 or args.val_ratio <= 0:
            raise ValueError("train_ratio and val_ratio must be > 0")
        total_ratio = args.train_ratio + args.val_ratio
        train_ratio = args.train_ratio / total_ratio

        records_all, convert_stats = _convert_payload(
            _read_json(args.ann_json), args.image_root, skip_missing_images=bool(args.skip_missing_images)
        )
        records = _apply_max_samples(records_all, args.max_samples)
        n_total = len(records)
        n_train = int(n_total * train_ratio)
        train_records = records[:n_train]
        val_records = records[n_train:]

        outputs = {}
        outputs.update(_save_split(output_dir, "train", train_records, args.emit_swift_format))
        outputs.update(_save_split(output_dir, "val", val_records, args.emit_swift_format))
        summary = {
            "mode": "single_json_split_by_ratio",
            "ann_json": args.ann_json,
            "image_root": args.image_root,
            "num_total": n_total,
            "num_train": len(train_records),
            "num_val": len(val_records),
            "convert_stats": convert_stats,
            **outputs,
        }
    else:
        raise ValueError("Provide either --train-ann-json + --val-ann-json, or --ann-json.")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
