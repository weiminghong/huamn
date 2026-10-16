from __future__ import annotations

import argparse
import json
from pathlib import Path

from .manifest import audit_manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    audit = audit_manifests(args.manifests)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, sort_keys=True))
    if not audit["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
