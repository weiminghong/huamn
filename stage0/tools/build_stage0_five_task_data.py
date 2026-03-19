import argparse
import os.path as osp
import random
import sys
from typing import Any, Dict, List, Tuple

THIS_DIR = osp.abspath(osp.dirname(__file__))
HUMAN_ROOT = osp.abspath(osp.join(THIS_DIR, "..", ".."))
PROJECT_ROOT = osp.abspath(osp.join(HUMAN_ROOT, ".."))
for p in (PROJECT_ROOT, HUMAN_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)


from common.five_task_builder import build_five_task_target_text
from common.io_utils import dump_jsonl, load_jsonl
from common.prompting import build_stage1_five_task_prompt



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage0 five-task JSONL for Stage1.")
    parser.add_argument("--input-jsonl", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--seg-token", type=str, default="<SEG>")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--prefer-split-field", action="store_true")
    parser.add_argument("--keep-order", action="store_true")
    parser.add_argument("--max-instances", type=int, default=30)
    return parser.parse_args()


def _normalize_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[float, float, float]:
    total = train_ratio + val_ratio + test_ratio
    if total <= 0:
        raise ValueError("train_ratio + val_ratio + test_ratio must be > 0")
    return train_ratio / total, val_ratio / total, test_ratio / total


def _convert_record(rec: Dict[str, Any], idx: int, seg_token: str, max_instances: int) -> Dict[str, Any]:
    target_text, mask_paths, payload = build_five_task_target_text(
        sample=rec,
        seg_token=seg_token,
        max_instances=max_instances,
    )
    prompt = build_stage1_five_task_prompt()
    image = rec.get("image")
    if not isinstance(image, str) or not image:
        raise ValueError(f"Record {idx} has no valid image field.")
    converted = {
        "id": rec.get("id", str(idx)),
        "image": image,
        "box": payload["box"],
        "pose": payload["pose"],
        "attribute": payload["attribute"],
        "relation": payload["relation"],
        "parsing": payload["parsing"],
        "mask_paths": mask_paths,
        "prompt": prompt,
        "target_text": target_text,
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": target_text},
        ],
    }
    if "split" in rec:
        converted["split"] = rec["split"]
    return converted


def _split_with_fixed_ratio(
    records: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
    keep_order: bool,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    idxs = list(range(len(records)))
    if not keep_order:
        rnd = random.Random(seed)
        rnd.shuffle(idxs)

    n = len(idxs)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    train = [records[i] for i in idxs[:n_train]]
    val = [records[i] for i in idxs[n_train : n_train + n_val]]
    test = [records[i] for i in idxs[n_train + n_val : n_train + n_val + n_test]]
    return train, val, test


def _split_from_field(records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    train, val, test = [], [], []
    for rec in records:
        split = str(rec.get("split", "")).lower()
        if split in ("train", "training"):
            train.append(rec)
        elif split in ("val", "valid", "validation", "dev"):
            val.append(rec)
        elif split in ("test", "testing"):
            test.append(rec)
        else:
            train.append(rec)
    return train, val, test


def main() -> None:
    args = parse_args()
    train_ratio, val_ratio, test_ratio = _normalize_ratios(
        args.train_ratio, args.val_ratio, args.test_ratio
    )

    records = load_jsonl(args.input_jsonl)
    if args.max_samples is not None:
        records = records[: args.max_samples]

    converted = []
    for i, rec in enumerate(records):
        converted.append(
            _convert_record(
                rec=rec,
                idx=i,
                seg_token=args.seg_token,
                max_instances=args.max_instances,
            )
        )

    if args.prefer_split_field and any("split" in x for x in converted):
        train, val, test = _split_from_field(converted)
    else:
        train, val, test = _split_with_fixed_ratio(
            records=converted,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=args.seed,
            keep_order=args.keep_order,
        )

    train_path = osp.join(args.output_dir, "train.jsonl")
    val_path = osp.join(args.output_dir, "val.jsonl")
    test_path = osp.join(args.output_dir, "test.jsonl")
    summary_path = osp.join(args.output_dir, "summary.json")

    dump_jsonl(train_path, train)
    dump_jsonl(val_path, val)
    dump_jsonl(test_path, test)

    summary = {
        "input_jsonl": args.input_jsonl,
        "output_dir": args.output_dir,
        "seg_token": args.seg_token,
        "num_total": len(converted),
        "num_train": len(train),
        "num_val": len(val),
        "num_test": len(test),
    }

    import json
    from pathlib import Path

    summary_file = Path(summary_path)
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
