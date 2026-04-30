from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage1 SFT data and a Stage2 manifest.")
    parser.add_argument("--train-ann-json", type=str, required=True)
    parser.add_argument("--val-ann-json", type=str, required=True)
    parser.add_argument("--train-image-root", type=str, required=True)
    parser.add_argument("--val-image-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--skip-missing-images", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--emit-swift-format", action="store_true", default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "qwen4b_coco_unihuman_ft" / "convert_coco_unihuman.py"),
        "--train-ann-json",
        str(args.train_ann_json),
        "--val-ann-json",
        str(args.val_ann_json),
        "--image-root",
        str(args.train_image_root),
        "--train-image-root",
        str(args.train_image_root),
        "--val-image-root",
        str(args.val_image_root),
        "--output-dir",
        str(out_dir),
    ]
    if args.max_samples is not None:
        cmd.extend(["--max-samples", str(args.max_samples)])
    cmd.append("--skip-missing-images" if args.skip_missing_images else "--no-skip-missing-images")
    if args.emit_swift_format:
        cmd.append("--emit-swift-format")

    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))

    manifest = {
        "train_ann_json": str(args.train_ann_json),
        "val_ann_json": str(args.val_ann_json),
        "train_image_root": str(args.train_image_root),
        "val_image_root": str(args.val_image_root),
        "stage1_train_jsonl": str(out_dir / "train.jsonl"),
        "stage1_val_jsonl": str(out_dir / "val.jsonl"),
        "stage1_train_swift_jsonl": str(out_dir / "train_swift.jsonl"),
        "stage1_val_swift_jsonl": str(out_dir / "val_swift.jsonl"),
    }
    with (out_dir / "stage12_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
