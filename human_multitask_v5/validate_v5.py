from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List

from .manifest import audit_manifests


MODEL_ROOTS = (
    Path("/root/autodl-tmp/model"),
    Path("/root/autodl-tmp/humanslot_refs"),
    Path("/root/autodl-tmp/hf_stage2/hub"),
    Path("/root/autodl-tmp/humanslot/HumanQueryNet_mmdet3_port/models"),
)


def model_inventory() -> List[str]:
    matches = []
    for root in MODEL_ROOTS:
        if not root.is_dir():
            continue
        for child in root.glob("*"):
            if "mask2former" in child.name.lower():
                matches.append(str(child.resolve()))
    return sorted(set(matches))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_v5.json")),
    )
    parser.add_argument("--manifests", nargs="*", default=[])
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    source_rows: Dict[str, Dict] = {}
    blockers = []
    for name, source in config["sources"].items():
        row = {"declared_status": source["status"]}
        if source["status"] == "available":
            paths = [
                value
                for key, value in source.items()
                if key.endswith("annotation")
                or key.endswith("image_root")
            ]
            missing = [value for value in paths if not Path(value).exists()]
            row["paths_checked"] = paths
            row["missing_paths"] = missing
            if missing:
                blockers.append(f"{name}:missing_declared_asset")
        else:
            row["missing_paths"] = [source.get("root", "unspecified")]
            blockers.append(f"{name}:source_assets_absent")
        source_rows[name] = row

    usage = shutil.disk_usage("/root/autodl-tmp")
    free_gib = usage.free / 2**30
    if free_gib < 20:
        blockers.append("data_disk_free_space_below_20GiB")
    inventory = model_inventory()
    if not inventory:
        blockers.append("mask2former_pretrained_checkpoint_absent")
    scorers = json.loads(
        Path(__file__).with_name("scorers_v5.json").read_text(
            encoding="utf-8"
        )
    )
    missing_scorers = sorted(
        name for name, command in scorers.items() if not command
    )
    if missing_scorers:
        blockers.append("official_scorer_adapters_incomplete")

    report = {
        "schema": "human-multitask-v5-readiness",
        "ready_for_coco_preflight": not source_rows[
            "coco_unihuman"
        ]["missing_paths"],
        "ready_for_multisource_training": not blockers,
        "sources": source_rows,
        "manifest_audit": (
            audit_manifests(args.manifests) if args.manifests else None
        ),
        "model_roots_audited": [str(path) for path in MODEL_ROOTS],
        "mask2former_inventory": inventory,
        "missing_official_scorers": missing_scorers,
        "data_disk_free_gib": round(free_gib, 3),
        "blockers": blockers,
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if args.require_ready and blockers:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
