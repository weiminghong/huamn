import argparse
import importlib.util
import json
import os.path as osp
import sys
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from human.common.five_task_metrics import evaluate_five_task_records
from human.common.io_utils import dump_jsonl
from human.common.stage1_five_task_dataset import Stage1FiveTaskDataset
from human.common.target_builder import parse_json_object_from_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage1 five-task evaluation")
    parser.add_argument("--config", type=str, default="configs/qwen_vl_stage1_five_tasks.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--output-metrics", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = osp.abspath(path)
    spec = importlib.util.spec_from_file_location("stage1_five_task_cfg", cfg_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load config from: {cfg_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "CONFIG"):
        raise AttributeError(f"CONFIG not found in {cfg_path}")
    return dict(module.CONFIG)


def get_torch_dtype(cfg: Dict[str, Any]):
    if cfg.get("bf16", False):
        return torch.bfloat16
    if cfg.get("fp16", False):
        return torch.float16
    return torch.float32


def build_model(model_path: str, torch_dtype):
    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            )
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch_dtype, trust_remote_code=True
    )


def collate_identity(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    split_jsonl = cfg["val_jsonl"] if args.split == "val" else cfg.get("test_jsonl", cfg["val_jsonl"])
    dataset = Stage1FiveTaskDataset(
        jsonl_path=split_jsonl,
        image_root=cfg.get("image_root", None),
        mask_root=cfg.get("mask_root", None),
        for_training=False,
        max_samples=args.max_samples if args.max_samples is not None else cfg.get("max_val_samples", None),
        extra_prompt=cfg.get("extra_prompt", None),
        seg_token=cfg.get("seg_token", "<SEG>"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("dataloader_num_workers", 2)),
        collate_fn=collate_identity,
    )

    ckpt = args.checkpoint if args.checkpoint is not None else cfg["output_dir"]
    processor_dir = osp.join(ckpt, "processor")
    proc_source = processor_dir if osp.isdir(processor_dir) else ckpt
    processor = AutoProcessor.from_pretrained(proc_source, trust_remote_code=True)
    model = build_model(ckpt, get_torch_dtype(cfg))
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    max_length = int(cfg.get("max_length", 4096))
    max_new_tokens = int(args.max_new_tokens)

    pred_lines: List[Dict[str, Any]] = []
    eval_records: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            prompts = [s["prompt"] + "\n" for s in batch]
            images = [s["image"] for s in batch]
            inputs = processor(
                text=prompts,
                images=images,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

            input_len = inputs["input_ids"].shape[1]
            gen_suffix = gen_ids[:, input_len:]
            pred_texts = processor.batch_decode(gen_suffix, skip_special_tokens=True)

            for sample, pred_text in zip(batch, pred_texts):
                pred_struct = parse_json_object_from_text(pred_text)
                valid_json = pred_struct is not None
                if pred_struct is None:
                    pred_struct = {
                        "box": [],
                        "pose": [],
                        "attribute": [],
                        "relation": [],
                        "parsing": [],
                    }

                gt_struct = parse_json_object_from_text(sample["target_text"])
                if gt_struct is None:
                    gt_struct = {
                        "box": [],
                        "pose": [],
                        "attribute": [],
                        "relation": [],
                        "parsing": [],
                    }

                pred_lines.append(
                    {
                        "id": sample["id"],
                        "image_path": sample["image_path"],
                        "prompt": sample["prompt"],
                        "prediction_text": pred_text,
                        "prediction_struct": pred_struct,
                        "valid_json": valid_json,
                    }
                )
                eval_records.append(
                    {
                        "id": sample["id"],
                        "valid_json": valid_json,
                        "pred_struct": pred_struct,
                        "gt_struct": gt_struct,
                    }
                )

    metrics = evaluate_five_task_records(eval_records)

    output_jsonl = (
        args.output_jsonl
        if args.output_jsonl is not None
        else osp.join(cfg["output_dir"], f"stage1_five_tasks_{args.split}_predictions.jsonl")
    )
    output_metrics = (
        args.output_metrics
        if args.output_metrics is not None
        else osp.join(cfg["output_dir"], f"stage1_five_tasks_{args.split}_metrics.json")
    )

    dump_jsonl(output_jsonl, pred_lines)
    with open(output_metrics, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
