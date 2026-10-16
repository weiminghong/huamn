from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_ROOT = REPO_ROOT / "COCO-UniHuman-main" / "HumanQueryNet"
DEFAULT_CONFIG = REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage2_qwen_ie_dinov3_hqnet.py"
DEFAULT_DINOV3_MODEL = "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m"
FUSION_MODES = ("feature_add", "feature_concat", "input_add", "input_replace", "none", "add", "replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Stage2 DINOv3/HumanQueryNet training.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--train-ann-json", type=str, required=True)
    parser.add_argument("--val-ann-json", type=str, required=True)
    parser.add_argument("--train-image-root", type=str, required=True)
    parser.add_argument("--val-image-root", type=str, required=True)
    parser.add_argument("--qwen-model-name-or-path", type=str, default="")
    parser.add_argument("--qwen-adapter-path", type=str, default="")
    parser.add_argument("--dino-model-name", type=str, default=DEFAULT_DINOV3_MODEL)
    parser.add_argument("--fusion-mode", type=str, default="feature_add", choices=FUSION_MODES)
    parser.add_argument("--freeze-dino", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--launcher", type=str, default="none", choices=["none", "pytorch", "slurm", "mpi"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config_name = Path(args.config).name
    is_dino_baseline = config_name == "stage2_dinov3_hqnet_baseline.py"
    if not is_dino_baseline and not args.qwen_model_name_or_path:
        parser.error("--qwen-model-name-or-path is required for Qwen-IE fusion Stage2 configs")
    return args


def main() -> None:
    args = parse_args()
    env = dict(os.environ)
    env["COCO_UNIHUMAN_TRAIN_JSON"] = str(args.train_ann_json)
    env["COCO_UNIHUMAN_VAL_JSON"] = str(args.val_ann_json)
    env["COCO_UNIHUMAN_TRAIN_IMG_ROOT"] = str(args.train_image_root)
    env["COCO_UNIHUMAN_VAL_IMG_ROOT"] = str(args.val_image_root)
    if args.qwen_model_name_or_path:
        env["STAGE1_QWEN_MODEL_NAME_OR_PATH"] = str(args.qwen_model_name_or_path)
    if args.qwen_adapter_path:
        env["STAGE1_QWEN_ADAPTER_PATH"] = str(args.qwen_adapter_path)
    env["DINOV3_MODEL_NAME"] = str(args.dino_model_name)
    env["STAGE2_WORK_DIR"] = str(args.work_dir)
    env["STAGE2_FUSION_MODE"] = str(args.fusion_mode)
    env["STAGE2_FREEZE_DINO"] = "1" if args.freeze_dino else "0"
    env["STAGE2_MAX_EPOCHS"] = str(args.max_epochs)

    py_path_parts = [str(REPO_ROOT), str(HQNET_ROOT)]
    old_py_path = env.get("PYTHONPATH", "")
    if old_py_path:
        py_path_parts.append(old_py_path)
    env["PYTHONPATH"] = os.pathsep.join(py_path_parts)

    cmd = [
        sys.executable,
        str(HQNET_ROOT / "tools" / "train.py"),
        str(args.config),
        "--work-dir",
        str(args.work_dir),
        "--gpu-id",
        str(args.gpu_id),
        "--seed",
        str(args.seed),
        "--launcher",
        str(args.launcher),
    ]
    if args.validate:
        cmd.append("--validate")
    subprocess.run(cmd, check=True, cwd=str(HQNET_ROOT), env=env)


if __name__ == "__main__":
    main()
