from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from PIL import Image

from human_slot_memory_pipeline.human_slot_memory.config import load_config
from human_slot_memory_pipeline.human_slot_memory.data.jsonl_dataset import HumanJsonlDataset
from human_slot_memory_pipeline.human_slot_memory.prompts import (
    DEFAULT_ASSISTANT_PREFIX_STAGE1,
    build_stage1_probe_prompt,
)


BOX_RE = re.compile(r"<box>\s*\[([^\]]+)\]\s*</box>", re.IGNORECASE | re.DOTALL)
POSE_RE = re.compile(r"<pose>\s*([\s\S]*?)\s*</pose>", re.IGNORECASE)
PERSON_RE = re.compile(r"<person\b[^>]*>([\s\S]*?)</person>", re.IGNORECASE)
TRIPLET_RE = re.compile(r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a 10-image Stage-1 prompt probe.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/prompt_debug10.yaml")
    parser.add_argument("--num-samples", type=int, default=None)
    return parser.parse_args()


def parse_float_list(text: str) -> List[float]:
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def parse_prediction_text(text: str) -> Dict[str, Any]:
    people: List[Dict[str, Any]] = []
    for block in PERSON_RE.findall(text):
        person: Dict[str, Any] = {}
        box_match = BOX_RE.search(block)
        if box_match:
            coords = parse_float_list(box_match.group(1))
            if len(coords) == 4:
                person["box_xyxy"] = coords
        pose_match = POSE_RE.search(block)
        if pose_match:
            pose = [[float(x), float(y), float(v)] for x, y, v in TRIPLET_RE.findall(pose_match.group(1))]
            if pose:
                person["pose17_xyv"] = pose
        if person:
            people.append(person)
    return {"people": people, "valid": len(people) > 0}


def maybe_denormalize_box(box: List[float], width: int, height: int) -> List[float]:
    if not box:
        return box
    if max(box) <= 1000.0 and width > 0 and height > 0:
        scale = [width / 1000.0, height / 1000.0, width / 1000.0, height / 1000.0]
        return [coord * factor for coord, factor in zip(box, scale)]
    return box


def maybe_denormalize_pose(pose: List[List[float]], width: int, height: int) -> List[List[float]]:
    if not pose:
        return pose
    max_coord = max((max(point[:2]) for point in pose if len(point) >= 2), default=0.0)
    if max_coord <= 1000.0 and width > 0 and height > 0:
        denorm = []
        for point in pose:
            if len(point) < 3:
                denorm.append(point)
                continue
            denorm.append([point[0] * width / 1000.0, point[1] * height / 1000.0, point[2]])
        return denorm
    return pose


def compute_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(ax2 - ax1, 0.0) * max(ay2 - ay1, 0.0)
    area_b = max(bx2 - bx1, 0.0) * max(by2 - by1, 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def greedy_match(gt_boxes: List[List[float]], pred_boxes: List[List[float]], thr: float = 0.5) -> List[Tuple[int, int, float]]:
    matches: List[Tuple[int, int, float]] = []
    used_gt = set()
    used_pred = set()
    candidates: List[Tuple[float, int, int]] = []
    for gt_idx, gt_box in enumerate(gt_boxes):
        for pred_idx, pred_box in enumerate(pred_boxes):
            iou = compute_iou(gt_box, pred_box)
            if iou >= thr:
                candidates.append((iou, gt_idx, pred_idx))
    for iou, gt_idx, pred_idx in sorted(candidates, reverse=True):
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        matches.append((gt_idx, pred_idx, iou))
    return matches


def pose_pck(gt_pose: List[List[float]], pred_pose: List[List[float]], gt_box: List[float], thr: float = 0.2) -> Tuple[int, int]:
    scale = max(gt_box[2] - gt_box[0], gt_box[3] - gt_box[1], 1.0)
    correct = 0
    total = 0
    for gt_point, pred_point in zip(gt_pose, pred_pose):
        if len(gt_point) < 3 or gt_point[2] <= 0:
            continue
        total += 1
        dist = math.sqrt((gt_point[0] - pred_point[0]) ** 2 + (gt_point[1] - pred_point[1]) ** 2)
        if dist <= thr * scale:
            correct += 1
    return correct, total


def load_model(model_path: str, adapter_path: str = "", use_bf16: bool = True):
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    model = AutoModelForImageTextToText.from_pretrained(model_path, trust_remote_code=True, torch_dtype=dtype)
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    return processor, model


def build_chat_text(processor: Any, prompt: str) -> str:
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + DEFAULT_ASSISTANT_PREFIX_STAGE1


def run_one(processor: Any, model: Any, image_path: str, prompt: str, max_length: int, max_new_tokens: int) -> str:
    image = Image.open(image_path).convert("RGB")
    text = build_chat_text(processor, prompt)
    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )
    device = next(model.parameters()).device
    inputs = {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
        )
    input_len = inputs["input_ids"].shape[1]
    pred_suffix = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)[0]
    return DEFAULT_ASSISTANT_PREFIX_STAGE1 + pred_suffix


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dir = cfg.ensure_output_dir()
    num_samples = int(args.num_samples or cfg.runtime.get("num_samples", 10))
    dataset = HumanJsonlDataset(cfg.paths["val_jsonl"], image_root=cfg.paths.get("image_root"))
    indices = list(range(len(dataset)))
    random.Random(int(cfg.project.get("seed", 42))).shuffle(indices)
    indices = indices[:num_samples]

    processor, model = load_model(
        str(cfg.paths["qwen_model"]),
        adapter_path=str(cfg.paths.get("adapter_path", "")),
        use_bf16=bool(cfg.optimization.get("bf16", True)),
    )
    model.eval().to("cuda")

    prompt = build_stage1_probe_prompt()
    rows: List[Dict[str, Any]] = []
    total_gt = 0
    total_pred = 0
    total_tp = 0
    pose_correct = 0
    pose_total = 0
    valid_count = 0

    for sample_index in indices:
        sample = dataset[sample_index]
        pred_text = run_one(
            processor=processor,
            model=model,
            image_path=sample.image_path,
            prompt=prompt,
            max_length=int(cfg.stage1.get("max_length", 2048)),
            max_new_tokens=int(cfg.stage1.get("max_new_tokens", 1024)),
        )
        parsed = parse_prediction_text(pred_text)
        gt_boxes = [slot.box_xyxy for slot in sample.slots if slot.box_xyxy]
        pred_boxes = [
            maybe_denormalize_box(person["box_xyxy"], sample.width, sample.height)
            for person in parsed["people"]
            if "box_xyxy" in person
        ]
        matches = greedy_match(gt_boxes, pred_boxes, thr=0.5)
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)
        total_tp += len(matches)
        valid_count += int(parsed["valid"])

        for gt_idx, pred_idx, _ in matches:
            gt_pose = sample.slots[gt_idx].pose17_xyv
            pred_pose = parsed["people"][pred_idx].get("pose17_xyv")
            if pred_pose:
                pred_pose = maybe_denormalize_pose(pred_pose, sample.width, sample.height)
            if gt_pose and pred_pose:
                correct, total = pose_pck(gt_pose, pred_pose, gt_boxes[gt_idx])
                pose_correct += correct
                pose_total += total

        rows.append(
            {
                "sample_id": sample.sample_id,
                "image_path": sample.image_path,
                "gt_boxes": gt_boxes,
                "prediction_text": pred_text,
                "parsed": parsed,
                "matches": [{"gt_index": g, "pred_index": p, "iou": iou} for g, p, iou in matches],
            }
        )

    metrics = {
        "num_samples": num_samples,
        "valid_rate": valid_count / max(num_samples, 1),
        "bbox_num_gt": total_gt,
        "bbox_num_pred": total_pred,
        "bbox_tp": total_tp,
        "bbox_precision": total_tp / max(total_pred, 1),
        "bbox_recall": total_tp / max(total_gt, 1),
        "pose_pck_on_matched": pose_correct / max(pose_total, 1),
        "pose_visible_total": pose_total,
    }

    with open(Path(output_dir) / "prompt_debug10_predictions.jsonl", "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(Path(output_dir) / "prompt_debug10_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
