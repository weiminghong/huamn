import argparse
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


DEFAULT_IMAGES_DIR = "/share2/program/weiminghong/coco/coco_person/images"
DEFAULT_LABELS_DIR = "/share2/program/weiminghong/coco/coco_person/labels"

DEFAULT_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_LABEL_EXTS = {".txt", ".json", ".xml", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Check whether image IDs and label IDs are one-to-one and whether counts are consistent."
        )
    )
    parser.add_argument("--images-dir", type=str, default=DEFAULT_IMAGES_DIR)
    parser.add_argument("--labels-dir", type=str, default=DEFAULT_LABELS_DIR)
    parser.add_argument(
        "--image-exts",
        type=str,
        default=",".join(sorted(DEFAULT_IMAGE_EXTS)),
        help="Comma-separated image extensions, e.g. .jpg,.png",
    )
    parser.add_argument(
        "--label-exts",
        type=str,
        default=",".join(sorted(DEFAULT_LABEL_EXTS)),
        help="Comma-separated label extensions, e.g. .txt,.json",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recursively scan subfolders (default: true).",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=20,
        help="How many mismatch examples to print.",
    )
    parser.add_argument(
        "--save-report",
        type=str,
        default=None,
        help="Optional path to save a JSON report.",
    )
    return parser.parse_args()


def parse_exts(ext_text: str) -> Set[str]:
    parts = [x.strip().lower() for x in ext_text.split(",") if x.strip()]
    exts: Set[str] = set()
    for ext in parts:
        if not ext.startswith("."):
            ext = f".{ext}"
        exts.add(ext)
    return exts


def collect_files(root: Path, exts: Set[str], recursive: bool) -> List[Path]:
    if recursive:
        files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]
    else:
        files = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in exts]
    files.sort()
    return files


def build_id_map(files: List[Path]) -> Dict[str, List[str]]:
    id_map: Dict[str, List[str]] = {}
    for p in files:
        sample_id = p.stem
        id_map.setdefault(sample_id, []).append(str(p))
    return id_map


def duplicate_ids(id_map: Dict[str, List[str]]) -> Dict[str, List[str]]:
    return {k: v for k, v in id_map.items() if len(v) > 1}


def summarize(
    image_files: List[Path],
    label_files: List[Path],
    image_id_map: Dict[str, List[str]],
    label_id_map: Dict[str, List[str]],
) -> Dict[str, object]:
    image_ids = set(image_id_map.keys())
    label_ids = set(label_id_map.keys())

    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)

    image_dup = duplicate_ids(image_id_map)
    label_dup = duplicate_ids(label_id_map)

    file_count_equal = len(image_files) == len(label_files)
    unique_id_count_equal = len(image_ids) == len(label_ids)
    one_to_one = (
        len(missing_labels) == 0
        and len(missing_images) == 0
        and len(image_dup) == 0
        and len(label_dup) == 0
        and unique_id_count_equal
    )

    passed = one_to_one and file_count_equal

    return {
        "passed": passed,
        "file_count_equal": file_count_equal,
        "unique_id_count_equal": unique_id_count_equal,
        "one_to_one_id_match": one_to_one,
        "num_image_files": len(image_files),
        "num_label_files": len(label_files),
        "num_unique_image_ids": len(image_ids),
        "num_unique_label_ids": len(label_ids),
        "num_missing_labels": len(missing_labels),
        "num_missing_images": len(missing_images),
        "num_duplicate_image_ids": len(image_dup),
        "num_duplicate_label_ids": len(label_dup),
        "missing_labels": missing_labels,
        "missing_images": missing_images,
        "duplicate_image_ids": image_dup,
        "duplicate_label_ids": label_dup,
    }


def print_report(report: Dict[str, object], topk: int) -> None:
    print("=== Dataset ID Match Report ===")
    print(f"passed: {report['passed']}")
    print(f"file_count_equal: {report['file_count_equal']}")
    print(f"unique_id_count_equal: {report['unique_id_count_equal']}")
    print(f"one_to_one_id_match: {report['one_to_one_id_match']}")
    print(f"num_image_files: {report['num_image_files']}")
    print(f"num_label_files: {report['num_label_files']}")
    print(f"num_unique_image_ids: {report['num_unique_image_ids']}")
    print(f"num_unique_label_ids: {report['num_unique_label_ids']}")
    print(f"num_missing_labels: {report['num_missing_labels']}")
    print(f"num_missing_images: {report['num_missing_images']}")
    print(f"num_duplicate_image_ids: {report['num_duplicate_image_ids']}")
    print(f"num_duplicate_label_ids: {report['num_duplicate_label_ids']}")

    missing_labels = report["missing_labels"]
    missing_images = report["missing_images"]
    duplicate_image_ids = report["duplicate_image_ids"]
    duplicate_label_ids = report["duplicate_label_ids"]

    if missing_labels:
        print(f"\nFirst {min(topk, len(missing_labels))} image IDs without labels:")
        for x in missing_labels[:topk]:
            print(f"  - {x}")
    if missing_images:
        print(f"\nFirst {min(topk, len(missing_images))} label IDs without images:")
        for x in missing_images[:topk]:
            print(f"  - {x}")
    if duplicate_image_ids:
        dup_keys = sorted(duplicate_image_ids.keys())
        print(f"\nFirst {min(topk, len(dup_keys))} duplicate image IDs:")
        for k in dup_keys[:topk]:
            print(f"  - {k}: {duplicate_image_ids[k]}")
    if duplicate_label_ids:
        dup_keys = sorted(duplicate_label_ids.keys())
        print(f"\nFirst {min(topk, len(dup_keys))} duplicate label IDs:")
        for k in dup_keys[:topk]:
            print(f"  - {k}: {duplicate_label_ids[k]}")


def main() -> int:
    args = parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    if not images_dir.exists():
        print(f"[ERROR] images dir not found: {images_dir}")
        return 1
    if not labels_dir.exists():
        print(f"[ERROR] labels dir not found: {labels_dir}")
        return 1

    image_exts = parse_exts(args.image_exts)
    label_exts = parse_exts(args.label_exts)

    image_files = collect_files(images_dir, image_exts, recursive=args.recursive)
    label_files = collect_files(labels_dir, label_exts, recursive=args.recursive)

    image_id_map = build_id_map(image_files)
    label_id_map = build_id_map(label_files)

    report = summarize(
        image_files=image_files,
        label_files=label_files,
        image_id_map=image_id_map,
        label_id_map=label_id_map,
    )
    print_report(report, topk=max(1, args.topk))

    if args.save_report:
        save_path = Path(args.save_report)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {save_path}")

    return 0 if bool(report["passed"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
