#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PKG_DIR = "/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs"
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

from human_lisa_qwen_v1.schema import (
    BOX_TOKEN,
    POSE_TOKEN,
    PROMPT,
    SPECIAL_TOKENS,
    GUIDED_JSON_SCHEMA,
    denorm_box_xyxy,
    denorm_pose_17,
    load_jsonl,
    parse_prediction_json,
    poses_by_pid,
    resolve_image_path,
    sorted_person_items,
    attrs_by_pid,
)
from human_lisa_qwen_v1.modeling import QwenLisaHeads, load_qwen_vl_backbone
from human_lisa_qwen_v1.train_lisa_qwen_v1 import add_special_tokens, hidden_size_from_config


def dtype_from_args(args) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def build_chat_text(processor, prompt: str) -> str:
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    try:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_eval_schema(allow_seg_token: bool = False) -> Dict[str, Any]:
    schema = json.loads(json.dumps(GUIDED_JSON_SCHEMA))
    if not allow_seg_token:
        person_props = schema["properties"]["persons"]["items"]["properties"]
        person_props["seg"] = {"type": "null"}
    return schema


def make_guided_fn(tokenizer, prompt_len: int, schema: Dict[str, Any]):
    parser = JsonSchemaParser(schema)
    base_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)
    def allowed(batch_id, sent):
        return base_fn(batch_id, sent[prompt_len:])
    return allowed


def iter_batches(rows: List[Dict[str, Any]], batch_size: int):
    batch_size = max(1, int(batch_size))
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def set_tokenizer_padding_side(processor, padding_side: str) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = padding_side


def empty_prediction() -> Dict[str, List[Any]]:
    return {"box": [], "pose": [], "attribute": []}


def error_result(exc: Exception, raw: str = "") -> Dict[str, Any]:
    return {"schema_ok": False, "raw": raw, "error": repr(exc), "prediction": empty_prediction()}


def decode_generated_json_text(tokenizer, token_ids: torch.Tensor) -> str:
    raw = tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ).strip()
    stop_tokens = [getattr(tokenizer, "eos_token", None), "<|im_end|>", "<|endoftext|>"]
    for stop in stop_tokens:
        if stop and stop in raw:
            raw = raw.split(stop, 1)[0]
    return raw.strip()


def iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(1e-9, aa + bb - inter)


def match_boxes(pred: List[Dict[str, Any]], gt: List[Dict[str, Any]], thr: float) -> List[Tuple[int, int, float]]:
    pairs = []
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gt):
            pairs.append((iou(p["bbox_xyxy"], g["bbox_xyxy"]), pi, gi))
    pairs.sort(reverse=True)
    used_p, used_g, out = set(), set(), []
    for ov, pi, gi in pairs:
        if ov >= thr and pi not in used_p and gi not in used_g:
            used_p.add(pi); used_g.add(gi); out.append((pi, gi, ov))
    return out


def gt_struct(record: Dict[str, Any]) -> Dict[str, Any]:
    width, height = int(record.get("width", 1)), int(record.get("height", 1))
    boxes = []
    for i, item in enumerate(sorted_person_items(record), start=1):
        boxes.append({"person_id": item.get("person_id", i), "bbox_xyxy": item.get("bbox_xyxy"), "score": 1.0})
    return {"boxes": boxes, "poses_by_pid": poses_by_pid(record), "attrs_by_pid": attrs_by_pid(record), "width": width, "height": height}


def assemble_prediction(obj: Dict[str, Any], decoded: Dict[str, Any], width: int, height: int) -> Dict[str, Any]:
    boxes_norm = list(decoded.get("boxes_norm", []))
    poses_norm = list(decoded.get("poses_norm", []))
    pred_boxes, pred_poses, pred_attrs = [], [], []
    box_i = pose_i = 0
    for idx, person in enumerate(obj.get("persons", []), start=1):
        score = float(person.get("score", 1.0)) if isinstance(person, dict) else 1.0
        pred_pid = int(person.get("person_id", idx)) if isinstance(person, dict) else idx
        if isinstance(person, dict) and person.get("box") == BOX_TOKEN and box_i < len(boxes_norm):
            pred_boxes.append({"person_id": pred_pid, "bbox_xyxy": denorm_box_xyxy(boxes_norm[box_i], width, height), "score": score})
            box_i += 1
        if isinstance(person, dict) and person.get("pose") == POSE_TOKEN and pose_i < len(poses_norm):
            pred_poses.append({"person_id": pred_pid, "keypoints": denorm_pose_17(poses_norm[pose_i], width, height), "score": score})
            pose_i += 1
        if isinstance(person, dict):
            attr = {"person_id": pred_pid}
            if int(person.get("gender", -1)) in (0, 1):
                attr["gender"] = int(person["gender"])
            if int(person.get("age", -1)) >= 0:
                attr["age"] = int(person["age"])
            if len(attr) > 1:
                pred_attrs.append(attr)
    return {"box": pred_boxes, "pose": pred_poses, "attribute": pred_attrs}


def pose_pck(pred_pose: List[List[float]], gt_pose: List[List[float]], gt_box: List[float], thr: float = 0.2) -> Tuple[int, int]:
    scale = max(1.0, ((gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])) ** 0.5)
    ok = total = 0
    for p, g in zip(pred_pose[:17], gt_pose[:17]):
        if len(g) < 3 or float(g[2]) <= 0:
            continue
        total += 1
        dx = float(p[0]) - float(g[0])
        dy = float(p[1]) - float(g[1])
        if (dx * dx + dy * dy) ** 0.5 <= thr * scale:
            ok += 1
    return ok, total


def decode_generated_heads_batch(wrapper, input_ids: torch.Tensor, hidden: torch.Tensor, start_positions: List[int]) -> List[Dict[str, Any]]:
    decoded: List[Dict[str, Any]] = []
    for b, start_pos in enumerate(start_positions):
        decoded.append(wrapper.decode_generated_heads(input_ids[b : b + 1], hidden[b : b + 1], int(start_pos)))
    return decoded


def evaluate_batch(wrapper, processor, rows: List[Dict[str, Any]], args, device) -> List[Dict[str, Any]]:
    if not rows:
        return []
    images = [Image.open(resolve_image_path(row, args.image_root)).convert("RGB") for row in rows]
    text = build_chat_text(processor, PROMPT)
    texts = [text] * len(rows)
    set_tokenizer_padding_side(processor, "left")
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(device)
    prompt_width = int(inputs["input_ids"].shape[-1])
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        prompt_start_positions = [int(x) for x in attention_mask.long().sum(dim=1).detach().cpu().tolist()]
    else:
        prompt_start_positions = [prompt_width] * len(rows)

    with torch.no_grad():
        gen = wrapper.base_model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            prefix_allowed_tokens_fn=make_guided_fn(processor.tokenizer, prompt_width, build_eval_schema(args.allow_seg_token)),
            pad_token_id=processor.tokenizer.eos_token_id,
        )

    results: List[Dict[str, Any]] = []
    valid_indices: List[int] = []
    for i in range(len(rows)):
        raw = decode_generated_json_text(processor.tokenizer, gen[i][prompt_width:])
        try:
            obj = parse_prediction_json(raw)
        except Exception as exc:
            results.append(error_result(exc, raw=raw))
            continue
        results.append({"schema_ok": True, "raw": raw, "obj": obj, "prediction": empty_prediction()})
        valid_indices.append(i)

    if valid_indices:
        valid_texts = [text + results[i]["raw"] for i in valid_indices]
        valid_images = [images[i] for i in valid_indices]
        # Right padding keeps the generated JSON positions aligned with the
        # prompt lengths used to select <box>/<pose>/<seg> hidden states.
        set_tokenizer_padding_side(processor, "right")
        head_inputs = processor(text=valid_texts, images=valid_images, return_tensors="pt", padding=True).to(device)
        vision_kwargs = {k: v for k, v in head_inputs.items() if k not in ("input_ids", "attention_mask")}
        start_positions = [prompt_start_positions[i] for i in valid_indices]
        with torch.no_grad():
            out = wrapper._forward_base(
                input_ids=head_inputs["input_ids"],
                attention_mask=head_inputs.get("attention_mask"),
                labels=None,
                **vision_kwargs,
            )
            decoded_items = decode_generated_heads_batch(wrapper, head_inputs["input_ids"], out.hidden_states[-1], start_positions)
        for idx, decoded in zip(valid_indices, decoded_items):
            row = rows[idx]
            results[idx]["prediction"] = assemble_prediction(
                results[idx]["obj"], decoded, int(row.get("width", 1)), int(row.get("height", 1))
            )
            results[idx].pop("obj", None)
    return results


def evaluate_one(wrapper, processor, row, args, device):
    result = evaluate_batch(wrapper, processor, [row], args, device)[0]
    if not result.get("schema_ok"):
        raise RuntimeError(result.get("error", "schema decode failed"))
    return result["raw"], result["prediction"]


def parse_args():
    ap = argparse.ArgumentParser(description="Legacy JSONL evaluator for older stage1_sft_data runs. For current COCO-UniHuman v1 official annotation evaluation, use eval_stage1_v1.py.")
    ap.add_argument("--jsonl", default="/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/val.jsonl")
    ap.add_argument("--image-root", default="")
    ap.add_argument("--model", default="/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--checkpoint", default="", help="Directory containing token_heads.pt or a token_heads.pt path.")
    ap.add_argument("--output-dir", default="/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/eval_debug")
    ap.add_argument("--max-samples", type=int, default=20)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=1, help="Number of images per VLM generate batch. 1 preserves the legacy behavior.")
    ap.add_argument("--cuda-visible-devices", default="0")
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--allow-seg-token", action=argparse.BooleanOptionalAction, default=False, help="Allow generated <seg> hooks in legacy JSONL eval. Current official-annotation v1 Stage1 uses eval_stage1_v1.py for segmentation.")
    ap.add_argument("--dry-run", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    rows = load_jsonl(args.jsonl)
    if args.max_samples is not None:
        rows = rows[: int(args.max_samples)]
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"prompt": PROMPT, "schema": GUIDED_JSON_SCHEMA, "num_rows": len(rows)}, ensure_ascii=False, indent=2))
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = Path(args.checkpoint) if args.checkpoint else None
    processor_path = ckpt / "processor" if ckpt is not None and (ckpt / "processor").is_dir() else Path(args.model)
    processor = AutoProcessor.from_pretrained(str(processor_path), trust_remote_code=True)
    base = load_qwen_vl_backbone(args.model, dtype_from_args(args))
    token_ids = add_special_tokens(processor, base)
    if ckpt is not None and (ckpt / "base_or_adapter").is_dir():
        try:
            from peft import PeftModel
            base = PeftModel.from_pretrained(base, str(ckpt / "base_or_adapter"))
        except Exception as e:
            raise RuntimeError(f"failed to load PEFT adapter from {ckpt / 'base_or_adapter'}: {e}") from e
    wrapper = QwenLisaHeads(base, hidden_size_from_config(base), token_ids).to(device)
    if args.checkpoint:
        head_path = ckpt if ckpt.name == "token_heads.pt" else ckpt / "token_heads.pt"
        wrapper.load_heads(str(head_path), map_location="cpu")
    else:
        print("WARNING: no --checkpoint provided; token heads are randomly initialized and metrics are only plumbing checks.", flush=True)
    wrapper.eval()
    set_tokenizer_padding_side(processor, "left")

    pred_path = out_dir / "predictions.jsonl"
    metrics_path = out_dir / "metrics.json"
    agg = {"images": 0, "schema_ok": 0, "schema_fail": 0, "gt_persons": 0, "pred_persons": 0, "matched": 0, "pose_ok": 0, "pose_total": 0, "gender_gt_matched": 0, "gender_answered": 0, "gender_correct_answered": 0, "age_gt_matched": 0, "age_answered": 0, "age_missing_when_gt_valid": 0}
    agg["age_abs_error_sum_answered"] = 0.0
    batch_size = max(1, int(args.batch_size))
    with pred_path.open("w", encoding="utf-8") as f:
        for batch_start in range(0, len(rows), batch_size):
            batch_rows = rows[batch_start : batch_start + batch_size]
            try:
                batch_results = evaluate_batch(wrapper, processor, batch_rows, args, device)
            except Exception as batch_exc:
                if batch_size > 1:
                    print(
                        json.dumps(
                            {"warning": "batch_eval_failed_falling_back_to_single", "batch_start": batch_start, "error": repr(batch_exc)},
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    batch_results = []
                    for row in batch_rows:
                        try:
                            batch_results.append(evaluate_batch(wrapper, processor, [row], args, device)[0])
                        except Exception as single_exc:
                            batch_results.append(error_result(single_exc))
                else:
                    batch_results = [error_result(batch_exc)]

            for offset, (row, result) in enumerate(zip(batch_rows, batch_results)):
                idx = batch_start + offset
                gt = gt_struct(row)
                rec = {"id": row.get("id"), "image": row.get("image")}
                pred = result.get("prediction", empty_prediction())
                if result.get("schema_ok"):
                    agg["schema_ok"] += 1
                    rec.update({"schema_ok": True, "raw": result.get("raw", ""), "prediction": pred})
                else:
                    agg["schema_fail"] += 1
                    rec.update({"schema_ok": False, "raw": result.get("raw", ""), "error": result.get("error", "unknown error"), "prediction": pred})
                matches = match_boxes(pred.get("box", []), gt["boxes"], args.iou_thr)
                pred_pose_by_pid = {p.get("person_id"): p for p in pred.get("pose", []) if isinstance(p, dict)}
                pred_attr_by_pid = {a.get("person_id"): a for a in pred.get("attribute", []) if isinstance(a, dict)}
                for pi, gi, ov in matches:
                    gt_pid = gt["boxes"][gi].get("person_id")
                    pred_pid = pred["box"][pi].get("person_id")
                    gp = gt["poses_by_pid"].get(gt_pid)
                    pp = pred_pose_by_pid.get(pred_pid)
                    if gp and pp:
                        ok, total = pose_pck(pp.get("keypoints", []), gp.get("keypoints", []), gt["boxes"][gi]["bbox_xyxy"])
                        agg["pose_ok"] += ok; agg["pose_total"] += total
                    ga = gt["attrs_by_pid"].get(gt_pid)
                    pa = pred_attr_by_pid.get(pred_pid)
                    if ga and pa:
                        if "gender" in ga:
                            agg["gender_gt_matched"] += 1
                            if "gender" in pa:
                                agg["gender_answered"] += 1
                                agg["gender_correct_answered"] += int(int(ga["gender"]) == int(pa["gender"]))
                        if "age" in ga:
                            agg["age_gt_matched"] += 1
                            if "age" in pa:
                                agg["age_answered"] += 1
                                agg["age_abs_error_sum_answered"] += abs(float(ga["age"]) - float(pa["age"]))
                            else:
                                agg["age_missing_when_gt_valid"] += 1
                agg["images"] += 1
                agg["gt_persons"] += len(gt["boxes"])
                agg["pred_persons"] += len(pred.get("box", []))
                agg["matched"] += len(matches)
                rec["matches"] = matches
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                if (idx + 1) % 20 == 0:
                    print(json.dumps({"processed": idx + 1, "schema_ok": agg["schema_ok"], "matched": agg["matched"]}, ensure_ascii=False), flush=True)
    metrics = dict(agg)
    metrics.update({
        "schema_ok_rate": agg["schema_ok"] / max(1, agg["images"]),
        "box_recall_iou50": agg["matched"] / max(1, agg["gt_persons"]),
        "box_precision_iou50": agg["matched"] / max(1, agg["pred_persons"]),
        "pose_pck@0.2": agg["pose_ok"] / max(1, agg["pose_total"]),
        "gender_coverage_on_matched_valid_gt": agg["gender_answered"] / max(1, agg["gender_gt_matched"]),
        "gender_acc_answered_only": agg["gender_correct_answered"] / max(1, agg["gender_answered"]),
        "gender_acc_unknown_as_wrong": agg["gender_correct_answered"] / max(1, agg["gender_gt_matched"]),
        "age_coverage_on_matched_valid_gt": agg["age_answered"] / max(1, agg["age_gt_matched"]),
        "age_mae_answered_only": agg["age_abs_error_sum_answered"] / max(1, agg["age_answered"]),
        "age_missing_rate_on_matched_valid_gt": agg["age_missing_when_gt_valid"] / max(1, agg["age_gt_matched"]),
        "task_scope": "Legacy stage1_sft_data JSONL evaluator. Current official-annotation v1 Stage1 metrics should be produced by eval_stage1_v1.py.",
        "seg_note": "This legacy JSONL file has no segmentation/parsing field. Official v1 annotation JSON has polygons and must be evaluated with eval_stage1_v1.py.",
    })
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
