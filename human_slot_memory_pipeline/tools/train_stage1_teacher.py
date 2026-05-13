from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from human_slot_memory_pipeline.human_slot_memory.config import load_config
from human_slot_memory_pipeline.human_slot_memory.stage1_data import (
    build_legacy_train_config,
    convert_raw_annotations_to_stage1_records,
    dump_yaml,
    save_stage1_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 structured MLLM teacher entry point.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/stage1_teacher.yaml")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.ensure_output_dir()

    train_records, train_stats = convert_raw_annotations_to_stage1_records(
        annotation_path=cfg.paths["train_ann_json"],
        image_root=cfg.paths["train_image_root"],
        max_samples=cfg.runtime.get("max_train_samples"),
        skip_missing_images=True,
    )
    val_records, val_stats = convert_raw_annotations_to_stage1_records(
        annotation_path=cfg.paths["val_ann_json"],
        image_root=cfg.paths["val_image_root"],
        max_samples=cfg.runtime.get("max_val_samples"),
        skip_missing_images=True,
    )

    data_dir = Path(output_dir) / "stage1_sft_data"
    train_jsonl = data_dir / "train.jsonl"
    val_jsonl = data_dir / "val.jsonl"
    save_stage1_jsonl(train_records, train_jsonl)
    save_stage1_jsonl(val_records, val_jsonl)

    runtime_overrides = {
        "seed": int(cfg.project.get("seed", 42)),
        "max_length": int(cfg.stage1.get("max_length", 2048)),
        "train_batch_size": int(cfg.optimization.get("batch_size", 1)),
        "eval_batch_size": int(cfg.optimization.get("eval_batch_size", cfg.optimization.get("batch_size", 1))),
        "gradient_accumulation_steps": int(cfg.optimization.get("gradient_accumulation_steps", 8)),
        "learning_rate": float(cfg.optimization.get("learning_rate", 1.0e-5)),
        "weight_decay": float(cfg.optimization.get("weight_decay", 0.01)),
        "num_train_epochs": float(cfg.stage1.get("num_train_epochs", cfg.optimization.get("num_train_epochs", 2))),
        "warmup_ratio": float(cfg.optimization.get("warmup_ratio", 0.03)),
        "bf16": bool(cfg.optimization.get("bf16", True)),
        "fp16": bool(cfg.optimization.get("fp16", False)),
        "gradient_checkpointing": bool(cfg.optimization.get("gradient_checkpointing", True)),
        "dataloader_num_workers": int(cfg.optimization.get("num_workers", 4)),
        "report_to": cfg.raw.get("report_to", "none"),
        "swanlab": cfg.raw.get("swanlab", {}),
        "lora": cfg.raw.get("lora", {}),
    }
    legacy_cfg = build_legacy_train_config(
        model_name_or_path=str(cfg.paths["qwen_model"]),
        train_jsonl=train_jsonl,
        val_jsonl=val_jsonl,
        image_root=str(cfg.paths["train_image_root"]),
        output_dir=output_dir,
        runtime_overrides=runtime_overrides,
    )
    legacy_cfg_path = Path(output_dir) / "stage1_legacy_sft.yaml"
    dump_yaml(legacy_cfg_path, legacy_cfg)

    summary = {
        "qwen_model": str(cfg.paths.get("qwen_model")),
        "output_dir": str(output_dir),
        "train_jsonl": str(train_jsonl),
        "val_jsonl": str(val_jsonl),
        "legacy_cfg": str(legacy_cfg_path),
        "train_stats": train_stats,
        "val_stats": val_stats,
    }
    with (Path(output_dir) / "stage1_prepare_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    should_run = bool(cfg.stage1.get("run_legacy_sft", True)) and not bool(args.prepare_only)
    if not should_run:
        return

    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) if not existing else str(repo_root) + os.pathsep + existing
    command = [sys.executable, "-m", "qwen4b_coco_unihuman_ft.train_sft", "--config", str(legacy_cfg_path)]
    subprocess.run(command, check=True, cwd=str(repo_root), env=env)


if __name__ == "__main__":
    main()
