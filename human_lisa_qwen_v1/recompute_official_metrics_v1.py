from __future__ import annotations

import argparse
import json
from pathlib import Path

from human_lisa_qwen_v1.official_eval_v1 import evaluate_all_official


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recompute COCO-UniHuman v1 official metrics from exported prediction records.")
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--attr-eval-source", default="/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port/datasets/api_wrappers/eval_attr.py")
    parser.add_argument("--merge-existing", default="", help="Optional existing metrics.json to merge non-official health fields from.")
    return parser.parse_args()


def load_predictions(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError("predictions must be a JSON list of COCO-style records")
    return data


def main() -> None:
    args = parse_args()
    predictions = load_predictions(args.predictions)
    image_ids = sorted({int(record["image_id"]) for record in predictions if "image_id" in record}) or None
    official = evaluate_all_official(
        args.annotation,
        predictions,
        image_ids=image_ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    payload = dict(official)
    payload.update({
        "annotation": args.annotation,
        "predictions": args.predictions,
        "recomputed_with": "human_lisa_qwen_v1.recompute_official_metrics_v1",
        "protocol": "official_cocoeval_and_humanquerynet_attr_recomputed",
    })
    if args.merge_existing:
        existing = json.loads(Path(args.merge_existing).read_text(encoding="utf-8"))
        health_keys = [
            "schema_ok", "schema_ok_rate", "health_ok", "health_ok_rate",
            "json_parse_ok_rate", "eos_seen_rate", "person_count_mae",
            "predicted_persons", "inference_file", "checkpoint", "manifest",
        ]
        for key in health_keys:
            if key in existing:
                payload[key] = existing[key]
    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
