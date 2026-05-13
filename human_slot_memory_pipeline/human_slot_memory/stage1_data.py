from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


def _read_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid annotation json: {path}")
    return payload


def _dump_jsonl(path: str | Path, records: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def dump_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)


def _is_valid_ann_for_field(annotation: Dict[str, Any], field: str, fallback_default: bool) -> bool:
    valid = annotation.get("valid", None)
    if isinstance(valid, dict):
        value = valid.get(field, 0)
        try:
            return bool(int(value))
        except Exception:
            return bool(value)
    return fallback_default


def _to_number(value: Any) -> Optional[float]:
    try:
        out = float(value)
    except Exception:
        return None
    if out != out:
        return None
    return out


def _to_int_if_possible(value: Any) -> Any:
    number = _to_number(value)
    if number is None:
        return value
    return int(number) if float(number).is_integer() else number


def _xywh_to_xyxy(bbox_xywh: List[Any]) -> Optional[List[float]]:
    if not isinstance(bbox_xywh, list) or len(bbox_xywh) != 4:
        return None
    x, y, w, h = [float(v) for v in bbox_xywh]
    if w <= 0 or h <= 0:
        return None
    return [x, y, x + w, y + h]


def _flat_kpts_to_triplets(flat_kpts: List[Any], max_keypoints: int = 17) -> Optional[List[List[float]]]:
    if not isinstance(flat_kpts, list) or len(flat_kpts) < 3:
        return None
    out: List[List[float]] = []
    count = min(max_keypoints, len(flat_kpts) // 3)
    for index in range(count):
        x = float(flat_kpts[index * 3 + 0])
        y = float(flat_kpts[index * 3 + 1])
        v = float(flat_kpts[index * 3 + 2])
        out.append([x, y, v])
    while len(out) < max_keypoints:
        out.append([0.0, 0.0, 0.0])
    return out[:max_keypoints]


def _norm_coord(value: float, size: int) -> int:
    if size <= 1:
        return 0
    clipped = max(0.0, min(float(value), float(size - 1)))
    return int(round(clipped / float(size) * 1000.0))


def _format_pose_triplets(keypoints: List[List[float]], width: int, height: int) -> str:
    triplets: List[str] = []
    padded = list(keypoints[:17])
    while len(padded) < 17:
        padded.append([0.0, 0.0, 0.0])
    for kp in padded[:17]:
        if not isinstance(kp, list) or len(kp) < 3 or float(kp[2]) <= 0:
            triplets.append("[0, 0, 0]")
            continue
        triplets.append(
            f"[{_norm_coord(float(kp[0]), width)}, {_norm_coord(float(kp[1]), height)}, {int(round(float(kp[2])))}]"
        )
    return ", ".join(triplets)


def build_stage1_target_text(record: Dict[str, Any]) -> str:
    width = int(record.get("width", 0) or 0)
    height = int(record.get("height", 0) or 0)
    poses_by_person = {
        item.get("person_id"): item.get("keypoints", [])
        for item in record.get("pose", [])
        if isinstance(item, dict)
    }
    attrs_by_person = {
        item.get("person_id"): item
        for item in record.get("attribute", [])
        if isinstance(item, dict)
    }
    person_blocks: List[str] = []
    for box_item in record.get("box", []):
        if not isinstance(box_item, dict):
            continue
        box = box_item.get("bbox_xyxy", [])
        if not isinstance(box, list) or len(box) != 4:
            continue
        person_id = box_item.get("person_id")
        attr = attrs_by_person.get(person_id, {})
        pose = poses_by_person.get(person_id, [[0.0, 0.0, 0.0] for _ in range(17)])
        box_norm = [
            _norm_coord(float(box[0]), width),
            _norm_coord(float(box[1]), height),
            _norm_coord(float(box[2]), width),
            _norm_coord(float(box[3]), height),
        ]
        gender = -1
        age = -1
        if isinstance(attr, dict) and "gender" in attr:
            gender = int(round(float(attr["gender"])))
        if isinstance(attr, dict) and "age" in attr:
            age = int(round(float(attr["age"])))
        person_blocks.append(
            "\n".join(
                [
                    "<person>",
                    f"<box>[{box_norm[0]}, {box_norm[1]}, {box_norm[2]}, {box_norm[3]}]</box>",
                    f"<pose>{_format_pose_triplets(pose, width, height)}</pose>",
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


def _extract_instances(annotations: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
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
    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        stats["num_instances"] += 1
        bbox = _xywh_to_xyxy(annotation.get("bbox", []))
        has_box = _is_valid_ann_for_field(annotation, "bboxes", fallback_default=bbox is not None)
        if not (has_box and bbox is not None):
            continue

        person_id += 1
        boxes.append({"person_id": person_id, "bbox_xyxy": bbox})
        stats["num_boxes"] += 1

        has_pose = _is_valid_ann_for_field(annotation, "keypoints_2d", fallback_default=False)
        keypoints = _flat_kpts_to_triplets(annotation.get("keypoints", []))
        if has_pose and keypoints is not None and any(point[2] > 0 for point in keypoints):
            poses.append({"person_id": person_id, "keypoints": keypoints})
            stats["num_poses"] += 1

        attr_item: Dict[str, Any] = {"person_id": person_id}
        has_gender = _is_valid_ann_for_field(annotation, "gender", fallback_default=("gender" in annotation))
        has_age = _is_valid_ann_for_field(annotation, "age", fallback_default=("age" in annotation))
        if has_gender and "gender" in annotation:
            attr_item["gender"] = _to_int_if_possible(annotation["gender"])
            stats["num_gender"] += 1
        if has_age and "age" in annotation:
            attr_item["age"] = _to_int_if_possible(annotation["age"])
            stats["num_age"] += 1
        if len(attr_item) > 1:
            attrs.append(attr_item)
            stats["num_attributes"] += 1
    return boxes, poses, attrs, stats


def convert_raw_annotations_to_stage1_records(
    annotation_path: str | Path,
    image_root: str | Path,
    max_samples: Optional[int] = None,
    skip_missing_images: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    payload = _read_json(annotation_path)
    images = payload.get("images", [])
    if not isinstance(images, list):
        raise ValueError("Invalid annotation payload: images must be a list.")
    if max_samples is not None:
        images = images[: int(max_samples)]

    anns_by_image: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        if not isinstance(annotation, dict):
            continue
        image_id = annotation.get("image_id")
        if image_id is None:
            continue
        try:
            anns_by_image[int(image_id)].append(annotation)
        except Exception:
            continue

    image_root_path = Path(image_root)
    records: List[Dict[str, Any]] = []
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
    for index, image in enumerate(images):
        if not isinstance(image, dict):
            continue
        stats["num_images_seen"] += 1
        image_id = int(image.get("id", index))
        file_name = str(image.get("file_name", "")).strip()
        if not file_name:
            continue
        image_abs = image_root_path / file_name
        if not image_abs.exists():
            stats["num_images_missing"] += 1
            if skip_missing_images:
                continue
        annotations = sorted(anns_by_image.get(image_id, []), key=lambda item: int(item.get("id", 0)))
        boxes, poses, attrs, image_stats = _extract_instances(annotations)
        for key, value in image_stats.items():
            stats[key] += value
        record = {
            "id": str(image_id),
            "image": file_name.replace("\\", "/"),
            "image_abs": str(image_abs),
            "width": int(image.get("width", 0) or 0),
            "height": int(image.get("height", 0) or 0),
            "box": boxes,
            "pose": poses,
            "attribute": attrs,
        }
        record["target_text"] = build_stage1_target_text(record)
        records.append(record)
        stats["num_images_kept"] += 1
    return records, stats


def save_stage1_jsonl(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    _dump_jsonl(output_path, records)


def build_legacy_train_config(
    model_name_or_path: str,
    train_jsonl: str | Path,
    val_jsonl: str | Path,
    image_root: str,
    output_dir: str | Path,
    runtime_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    config: Dict[str, Any] = {
        "model_name_or_path": str(model_name_or_path),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "image_root": str(image_root),
        "output_dir": str(output_dir / "legacy_sft"),
        "seed": 42,
        "max_train_samples": None,
        "max_val_samples": None,
        "max_length": 2048,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "learning_rate": 1.0e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 2,
        "warmup_ratio": 0.03,
        "logging_steps": 10,
        "save_steps": 200,
        "eval_steps": 200,
        "save_total_limit": 2,
        "logging_first_step": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "bf16": True,
        "fp16": False,
        "gradient_checkpointing": True,
        "dataloader_num_workers": 4,
        "report_to": "none",
        "extra_prompt": None,
        "swanlab": {
            "enable": True,
            "project": "human-slot-memory-stage1",
            "workspace": "wmh",
            "experiment_name": "human-slot-memory-stage1",
            "mode": "cloud",
            "preview_samples": 4,
            "preview_max_new_tokens": 1024,
            "stage_eval": {
                "enable": True,
                "samples": 16,
                "seed": 42,
                "iou_thr": 0.5,
                "pck_thr": 0.2,
            },
        },
        "lora": {
            "enable": True,
            "r": 8,
            "alpha": 32,
            "dropout": 0.05,
            "freeze_llm": False,
            "freeze_vit": True,
            "freeze_aligner": True,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        },
    }
    for key, value in (runtime_overrides or {}).items():
        config[key] = value
    return config
