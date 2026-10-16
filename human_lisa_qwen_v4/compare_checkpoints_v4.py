from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", required=True)
    parser.add_argument("--second", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def compare_values(
    first: Any,
    second: Any,
    *,
    path: str,
    report: Dict[str, Any],
) -> None:
    if torch.is_tensor(first) or torch.is_tensor(second):
        if not (torch.is_tensor(first) and torch.is_tensor(second)):
            report["differences"].append(f"{path}: tensor/type mismatch")
            return
        report["tensor_count"] += 1
        if first.shape != second.shape or first.dtype != second.dtype:
            report["differences"].append(
                f"{path}: {tuple(first.shape)}/{first.dtype} != "
                f"{tuple(second.shape)}/{second.dtype}"
            )
            return
        if not torch.equal(first.cpu(), second.cpu()):
            delta = (first.float().cpu() - second.float().cpu()).abs()
            maximum = float(delta.max().item()) if delta.numel() else 0.0
            report["max_abs_difference"] = max(
                report["max_abs_difference"], maximum
            )
            report["differences"].append(
                f"{path}: tensor values differ, max_abs={maximum}"
            )
        return
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        if not (isinstance(first, np.ndarray) and isinstance(second, np.ndarray)):
            report["differences"].append(f"{path}: ndarray/type mismatch")
        elif first.dtype != second.dtype or first.shape != second.shape:
            report["differences"].append(f"{path}: ndarray metadata differs")
        elif not np.array_equal(first, second):
            report["differences"].append(f"{path}: ndarray values differ")
        return
    if isinstance(first, dict) or isinstance(second, dict):
        if not (isinstance(first, dict) and isinstance(second, dict)):
            report["differences"].append(f"{path}: dict/type mismatch")
            return
        if set(first) != set(second):
            report["differences"].append(
                f"{path}: keys differ: "
                f"{sorted(set(first) ^ set(second), key=str)}"
            )
        for key in sorted(set(first) & set(second), key=str):
            compare_values(
                first[key], second[key], path=f"{path}.{key}", report=report
            )
        return
    if isinstance(first, (list, tuple)) or isinstance(second, (list, tuple)):
        if not (
            isinstance(first, (list, tuple))
            and isinstance(second, (list, tuple))
        ):
            report["differences"].append(f"{path}: sequence/type mismatch")
            return
        if len(first) != len(second):
            report["differences"].append(
                f"{path}: sequence lengths {len(first)} != {len(second)}"
            )
        for index, (left, right) in enumerate(zip(first, second)):
            compare_values(
                left, right, path=f"{path}[{index}]", report=report
            )
        return
    if first != second:
        report["differences"].append(f"{path}: {first!r} != {second!r}")


def load_components(checkpoint: Path) -> Dict[str, Any]:
    training = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    training = {
        key: value
        for key, value in training.items()
        if key != "elapsed_seconds"
    }
    return {
        "adapter": load_file(
            checkpoint / "base_or_adapter" / "adapter_model.safetensors",
            device="cpu",
        ),
        "merger": torch.load(
            checkpoint / "visual_merger.pt",
            map_location="cpu",
            weights_only=True,
        ),
        "tasks": torch.load(
            checkpoint / "stage1_v4_tasks.pt",
            map_location="cpu",
            weights_only=False,
        ),
        "training": training,
    }


def main() -> None:
    args = parse_args()
    first = Path(args.first).resolve()
    second = Path(args.second).resolve()
    report: Dict[str, Any] = {
        "first": str(first),
        "second": str(second),
        "exact": False,
        "tensor_count": 0,
        "max_abs_difference": 0.0,
        "differences": [],
    }
    compare_values(
        load_components(first),
        load_components(second),
        path="checkpoint",
        report=report,
    )
    report["exact"] = not report["differences"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["exact"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
