import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _maybe_load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Please install with: pip install pyyaml"
        ) from e
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Stage0 box+pose eval jsonl from person_with_instances.json + split txt files."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config path. CLI args override YAML values.",
    )
    parser.add_argument(
        "--ann-json",
        type=str,
        required=False,
        default="",
        help="Path to person_with_instances.json",
    )
    parser.add_argument(
        "--train-txt",
        type=str,
        required=False,
        default="",
        help="Path to train.txt (one image id/name per line)",
    )
    parser.add_argument(
        "--val-txt",
        type=str,
        required=False,
        default="",
        help="Path to val.txt (one image id/name per line)",
    )
    parser.add_argument(
        "--output-train-jsonl",
        type=str,
        required=False,
        default="",
        help="Output jsonl path for train split",
    )
    parser.add_argument(
        "--output-val-jsonl",
        type=str,
        required=False,
        default="",
        help="Output jsonl path for val split",
    )
    parser.add_argument(
        "--skip-empty-images",
        action="store_true",
        help="Skip images that have zero valid boxes and zero valid poses after filtering.",
    )
    args = parser.parse_args()

    if args.config:
        provided = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                provided.add(tok[2:].split("=")[0].replace("-", "_"))
        cfg = _maybe_load_yaml(args.config)
        for k, v in cfg.items():
            if not hasattr(args, k):
                continue
            if k in provided:
                continue
            setattr(args, k, v)

    return args


def load_split_tokens(path: str) -> List[str]:
    tokens: List[str] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                tokens.append(t)
    return tokens


def _stem_of(name: str) -> str:
    return Path(name).stem


def _xywh_to_xyxy(bbox: Sequence[Any]) -> Optional[List[float]]:
    if len(bbox) != 4:
        return None
    try:
        x, y, w, h = [float(v) for v in bbox]
    except Exception:
        return None
    x1 = x
    y1 = y
    x2 = x + w
    y2 = y + h
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _flat_kpts_to_triplets(flat_kpts: Sequence[Any]) -> List[List[float]]:
    out: List[List[float]] = []
    n = len(flat_kpts) // 3
    for i in range(n):
        try:
            x = float(flat_kpts[3 * i + 0])
            y = float(flat_kpts[3 * i + 1])
            v = float(flat_kpts[3 * i + 2])
        except Exception:
            continue
        out.append([x, y, v])
    return out


def _build_match_keys(rec: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()
    file_name = str(rec.get("file_name", "")).strip()
    if file_name:
        keys.add(file_name)
        keys.add(_stem_of(file_name))
    image_id = rec.get("id", None)
    if image_id is not None:
        keys.add(str(image_id))
    return keys


def _dump_jsonl(path: str, records: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def convert_record(image_rec: Dict[str, Any]) -> Dict[str, Any]:
    boxes: List[Dict[str, Any]] = []
    poses: List[Dict[str, Any]] = []

    for ins in image_rec.get("instance", []):
        if not isinstance(ins, dict):
            continue

        bbox = ins.get("bbox", [])
        if isinstance(bbox, list):
            xyxy = _xywh_to_xyxy(bbox)
            if xyxy is not None:
                boxes.append({"bbox_xyxy": xyxy})

        # Pose: keep only valid-person entries for evaluation
        num_keypoints = int(ins.get("num_keypoints", 0) or 0)
        flat_kpts = ins.get("keypoints", [])
        if num_keypoints > 0 and isinstance(flat_kpts, list):
            triplets = _flat_kpts_to_triplets(flat_kpts)
            if len(triplets) > 0:
                poses.append({"keypoints": triplets})

    return {
        "id": str(image_rec.get("id", "")),
        "image": str(image_rec.get("file_name", "")),
        "box": boxes,
        "pose": poses,
    }


def build_split(
    all_images: List[Dict[str, Any]],
    split_tokens: List[str],
    skip_empty_images: bool,
) -> Tuple[List[Dict[str, Any]], int]:
    token_set = set(split_tokens)
    matched = []
    used = 0

    for rec in all_images:
        keys = _build_match_keys(rec)
        if not keys:
            continue
        if len(keys & token_set) == 0:
            continue
        used += 1
        out = convert_record(rec)
        if skip_empty_images and len(out["box"]) == 0 and len(out["pose"]) == 0:
            continue
        matched.append(out)
    return matched, used


def main() -> None:
    args = parse_args()
    required_fields = [
        "ann_json",
        "train_txt",
        "val_txt",
        "output_train_jsonl",
        "output_val_jsonl",
    ]
    missing = [f for f in required_fields if not str(getattr(args, f, "")).strip()]
    if missing:
        raise ValueError(f"Missing required args: {missing}. Provide via CLI or --config YAML.")

    with Path(args.ann_json).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    all_images = payload.get("images", [])
    if not isinstance(all_images, list):
        raise ValueError(f"Invalid format: images must be list in {args.ann_json}")

    train_tokens = load_split_tokens(args.train_txt)
    val_tokens = load_split_tokens(args.val_txt)

    train_records, train_used = build_split(
        all_images=all_images,
        split_tokens=train_tokens,
        skip_empty_images=bool(args.skip_empty_images),
    )
    val_records, val_used = build_split(
        all_images=all_images,
        split_tokens=val_tokens,
        skip_empty_images=bool(args.skip_empty_images),
    )

    _dump_jsonl(args.output_train_jsonl, train_records)
    _dump_jsonl(args.output_val_jsonl, val_records)

    summary = {
        "ann_json": args.ann_json,
        "train_txt": args.train_txt,
        "val_txt": args.val_txt,
        "num_images_in_ann": len(all_images),
        "num_tokens_train": len(train_tokens),
        "num_tokens_val": len(val_tokens),
        "num_matched_train_before_filter": train_used,
        "num_matched_val_before_filter": val_used,
        "num_output_train": len(train_records),
        "num_output_val": len(val_records),
        "skip_empty_images": bool(args.skip_empty_images),
        "output_train_jsonl": args.output_train_jsonl,
        "output_val_jsonl": args.output_val_jsonl,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
