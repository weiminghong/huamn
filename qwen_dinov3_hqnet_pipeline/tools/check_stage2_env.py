from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path
from typing import Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_ROOT = REPO_ROOT / "COCO-UniHuman-main" / "HumanQueryNet"
DEFAULT_DINOV3_MODEL = "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether the current Python environment can run Stage2.")
    parser.add_argument("--config", default=str(REPO_ROOT / "qwen_dinov3_hqnet_pipeline" / "configs" / "stage2_qwen_ie_dinov3_hqnet.py"))
    parser.add_argument("--train-ann-json", default=os.getenv("COCO_UNIHUMAN_TRAIN_JSON", ""))
    parser.add_argument("--val-ann-json", default=os.getenv("COCO_UNIHUMAN_VAL_JSON", ""))
    parser.add_argument("--train-image-root", default=os.getenv("COCO_UNIHUMAN_TRAIN_IMG_ROOT", ""))
    parser.add_argument("--val-image-root", default=os.getenv("COCO_UNIHUMAN_VAL_IMG_ROOT", ""))
    parser.add_argument("--qwen-model-name-or-path", default=os.getenv("STAGE1_QWEN_MODEL_NAME_OR_PATH", ""))
    parser.add_argument("--dino-model-name", default=os.getenv("DINOV3_MODEL_NAME", DEFAULT_DINOV3_MODEL))
    parser.add_argument("--skip-paths", action="store_true", help="Only check imports and PYTHONPATH.")
    return parser.parse_args()


def _module_version(module_name: str) -> Tuple[bool, str]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return False, f"MISSING ({type(exc).__name__}: {exc})"
    return True, str(getattr(module, "__version__", "ok"))


def _check_mmcv_legacy_api() -> Tuple[bool, str]:
    ok, version = _module_version("mmcv")
    if not ok:
        return False, version
    try:
        from mmcv.utils import print_log  # noqa: F401
    except Exception as exc:
        return False, f"{version}; missing mmcv.utils.print_log ({type(exc).__name__}: {exc})"
    return True, version


def _check_pythonpath() -> Tuple[bool, str]:
    required = [str(REPO_ROOT), str(HQNET_ROOT)]
    missing = [path for path in required if path not in sys.path]
    if missing:
        return False, "missing from sys.path: " + ", ".join(missing)
    return True, "ok"


def _check_path(label: str, value: str, required: bool = True) -> Tuple[bool, str]:
    if not value:
        return (not required), "not provided"
    path = Path(value)
    if path.exists():
        return True, str(path)
    return False, f"does not exist: {path}"


def _print_result(label: str, ok: bool, detail: str) -> bool:
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}: {detail}")
    return ok


def main() -> int:
    args = parse_args()
    checks: List[Tuple[str, bool, str]] = []

    ok, detail = _module_version("torch")
    checks.append(("torch", ok, detail))
    ok, detail = _check_mmcv_legacy_api()
    checks.append(("mmcv legacy API", ok, detail))
    ok, detail = _module_version("mmdet")
    checks.append(("mmdet", ok, detail))
    ok, detail = _module_version("transformers")
    checks.append(("transformers", ok, detail))
    ok, detail = _check_pythonpath()
    checks.append(("PYTHONPATH", ok, detail))

    if not args.skip_paths:
        path_checks: Iterable[Tuple[str, str, bool]] = [
            ("config", args.config, True),
            ("COCO_UNIHUMAN_TRAIN_JSON", args.train_ann_json, True),
            ("COCO_UNIHUMAN_VAL_JSON", args.val_ann_json, True),
            ("COCO_UNIHUMAN_TRAIN_IMG_ROOT", args.train_image_root, True),
            ("COCO_UNIHUMAN_VAL_IMG_ROOT", args.val_image_root, True),
            ("STAGE1_QWEN_MODEL_NAME_OR_PATH", args.qwen_model_name_or_path, False),
            ("DINOV3_MODEL_NAME", args.dino_model_name, True),
        ]
        for label, value, required in path_checks:
            ok, detail = _check_path(label, value, required=required)
            checks.append((label, ok, detail))

    all_ok = True
    for label, ok, detail in checks:
        all_ok = _print_result(label, ok, detail) and all_ok

    if not all_ok:
        print("\nStage2 is not runnable in this environment yet. Use an environment with HumanQueryNet-compatible mmcv/mmdet plus transformers, and set PYTHONPATH to:")
        print(f"  export PYTHONPATH={REPO_ROOT}:{HQNET_ROOT}:$PYTHONPATH")
        return 1
    print("\nStage2 environment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
