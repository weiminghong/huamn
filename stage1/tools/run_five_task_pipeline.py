import argparse
import os
import os.path as osp
import shlex
import subprocess
import sys
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run stage0 + stage1 five-task full pipeline.")
    parser.add_argument("--unified-jsonl", type=str, required=True)
    parser.add_argument("--stage0-output-dir", type=str, default="data/human/stage1_five_tasks")
    parser.add_argument("--stage1-config", type=str, default="stage1/configs/qwen_vl_stage1_five_tasks.py")
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--mask-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="Outputs/stage1/qwen_vl_3b_five_tasks")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--skip-stage0", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    return parser.parse_args()


def _run(command: List[str], cwd: str) -> None:
    print(f"[RUN] {' '.join(shlex.quote(x) for x in command)}")
    proc = subprocess.run(command, cwd=cwd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(command)}")


def main() -> None:
    args = parse_args()
    repo_root = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
    python_exe = sys.executable

    stage0_train = osp.join(args.stage0_output_dir, "train.jsonl")
    stage0_val = osp.join(args.stage0_output_dir, "val.jsonl")

    if not args.skip_stage0:
        cmd = [
            python_exe,
            "stage0/tools/build_stage0_five_task_data.py",
            "--input-jsonl",
            args.unified_jsonl,
            "--output-dir",
            args.stage0_output_dir,
        ]
        if args.max_samples is not None:
            cmd += ["--max-samples", str(args.max_samples)]
        _run(cmd, cwd=repo_root)

    if not args.skip_train:
        cmd = [
            python_exe,
            "stage1/tools/train_five_tasks.py",
            "--config",
            args.stage1_config,
            "--train-jsonl",
            stage0_train,
            "--val-jsonl",
            stage0_val,
            "--output-dir",
            args.output_dir,
        ]
        if args.image_root:
            cmd += ["--image-root", args.image_root]
        if args.mask_root:
            cmd += ["--mask-root", args.mask_root]
        _run(cmd, cwd=repo_root)

    if not args.skip_test:
        cmd = [
            python_exe,
            "stage1/tools/test_five_tasks.py",
            "--config",
            args.stage1_config,
            "--checkpoint",
            args.output_dir,
            "--split",
            "val",
        ]
        if args.max_eval_samples is not None:
            cmd += ["--max-samples", str(args.max_eval_samples)]
        _run(cmd, cwd=repo_root)

    print("[DONE] five-task pipeline finished.")


if __name__ == "__main__":
    main()
