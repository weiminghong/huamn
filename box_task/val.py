import argparse
import importlib.util
import json
import os
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

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from box_task.dataset import BoxTxtDataset
from box_task.io_utils import dump_jsonl, parse_json_object_from_text
from box_task.metrics import evaluate_box_records
from box_task.postprocess import extract_pred_boxes, nms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Qwen-VL box-only structured generation.")
    parser.add_argument("--config", type=str, default="box_task/configs/box_qwenvl_data.py")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--score-thr", type=float, default=None)
    parser.add_argument("--nms-iou-thr", type=float, default=None)
    parser.add_argument("--max-per-img", type=int, default=None)
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    cfg_path = osp.abspath(path)
    spec = importlib.util.spec_from_file_location("box_task_cfg", cfg_path)
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
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


def collate_identity(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    split_cfg = dict(cfg["data"][args.split])

    dataset = BoxTxtDataset(
        image_dir=split_cfg["image_dir"],
        label_dir=split_cfg["label_dir"],
        list_file=split_cfg.get("list_file"),
        for_training=False,
        max_samples=args.max_samples,
        extra_prompt=cfg.get("extra_prompt"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.get("eval_batch_size", 1)),
        shuffle=False,
        num_workers=int(cfg.get("dataloader_num_workers", 2)),
        collate_fn=collate_identity,
    )

    ckpt = args.checkpoint if args.checkpoint is not None else cfg["model_name_or_path"]
    processor = AutoProcessor.from_pretrained(ckpt, trust_remote_code=True)
    model = build_model(ckpt, get_torch_dtype(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    output_dir = args.output_dir if args.output_dir is not None else cfg["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    score_thr = args.score_thr if args.score_thr is not None else float(cfg.get("score_thr", 0.0))
    nms_iou_thr = (
        args.nms_iou_thr if args.nms_iou_thr is not None else float(cfg.get("nms_iou_thr", 0.5))
    )
    max_per_img = args.max_per_img if args.max_per_img is not None else int(cfg.get("max_per_img", 100))
    eval_iou_thr = float(cfg.get("eval_iou_thr", 0.5))
    max_length = int(cfg.get("max_length", 4096))
    max_new_tokens = int(cfg.get("eval_max_new_tokens", cfg.get("max_new_tokens", 512)))

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
                    pred_struct = {"box": []}

                image_w, image_h = sample["image"].size
                pred_boxes = extract_pred_boxes(
                    pred_struct,
                    score_thr=score_thr,
                    image_size=[image_w, image_h],
                )
                pred_boxes = nms(pred_boxes, iou_thr=nms_iou_thr, max_per_img=max_per_img)
                pred_struct_post = {"box": pred_boxes}
                gt_struct = {"box": sample["labels"]["box"]}

                pred_lines.append(
                    {
                        "id": sample["id"],
                        "image_path": sample["image_path"],
                        "label_path": sample["label_path"],
                        "prompt": sample["prompt"],
                        "prediction_text": pred_text,
                        "prediction_struct_raw": pred_struct,
                        "prediction_struct_post": pred_struct_post,
                        "valid_json": valid_json,
                    }
                )
                eval_records.append(
                    {
                        "id": sample["id"],
                        "valid_json": valid_json,
                        "pred_struct": pred_struct_post,
                        "gt_struct": gt_struct,
                    }
                )

    metrics = evaluate_box_records(eval_records, iou_thr=eval_iou_thr)

    dump_jsonl(osp.join(output_dir, f"{args.split}_predictions.jsonl"), pred_lines)
    with open(osp.join(output_dir, f"{args.split}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
