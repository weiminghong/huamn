from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval_protocols import build_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--scorer-config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()

    protocols = build_plan({args.dataset: args.tasks})
    scorers = json.loads(
        Path(args.scorer_config).read_text(encoding="utf-8")
    )
    rows = []
    missing = []
    for protocol in protocols:
        command = scorers.get(protocol.scorer)
        row = {
            **protocol.__dict__,
            "command": command,
            "status": "ready" if command else "missing_official_scorer",
        }
        rows.append(row)
        if not command:
            missing.append(protocol.scorer)
    report = {
        "schema": "human-multitask-v5-evaluation-plan",
        "protocols": rows,
        "valid": not missing,
        "missing_scorers": sorted(set(missing)),
    }
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    if missing:
        raise SystemExit(2)
    if not args.preflight_only:
        raise RuntimeError(
            "prediction generation and scorer execution require a v5 checkpoint; "
            "preflight is the only admitted action before training"
        )


if __name__ == "__main__":
    main()
