import argparse
import os.path as osp
import sys
from typing import Any, Dict, List

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from human.common import build_stage1_target, dump_jsonl, load_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Build stage1 jsonl from unified labels")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-jsonl", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    records = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    output: List[Dict[str, Any]] = []
    for i, rec in enumerate(records):
        item = {
            "id": rec.get("id", str(i)),
            "image": rec["image"],
            "box": rec.get("box", []),
            "pose": rec.get("pose", []),
            "parsing": rec.get("parsing", []),
        }
        item["target_text"] = build_stage1_target(item)
        output.append(item)

    dump_jsonl(args.output_jsonl, output)
    print(f"Saved {len(output)} samples to: {args.output_jsonl}")


if __name__ == "__main__":
    main()
