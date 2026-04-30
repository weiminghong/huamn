from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage1_qwen25vl_lora.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Stage1 Qwen-VL SFT using the integrated pipeline config.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-name-or-path", type=str, default="")
    parser.add_argument("--train-jsonl", type=str, default="")
    parser.add_argument("--val-jsonl", type=str, default="")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--swanlab", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid stage1 yaml: {args.config}")

    overrides = {
        "model_name_or_path": args.model_name_or_path,
        "train_jsonl": args.train_jsonl,
        "val_jsonl": args.val_jsonl,
        "image_root": args.image_root,
        "output_dir": args.output_dir,
    }
    for key, value in overrides.items():
        if value:
            cfg[key] = value
    if args.swanlab is not None:
        cfg.setdefault("swanlab", {})
        cfg["swanlab"]["enable"] = bool(args.swanlab)

    out_dir = Path(str(cfg["output_dir"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_cfg = out_dir / "stage1_resolved.yaml"
    with resolved_cfg.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "qwen4b_coco_unihuman_ft" / "train_sft.py"),
        "--config",
        str(resolved_cfg),
    ]
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


if __name__ == "__main__":
    main()
