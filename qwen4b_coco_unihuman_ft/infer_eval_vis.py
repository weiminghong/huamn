import argparse
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
except Exception:
    Qwen3_5ForConditionalGeneration = None

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen4b_coco_unihuman_ft.eval_scheme2 import (  # noqa: E402
    dump_json,
    evaluate_records_scheme2,
    get_record_gt,
    load_jsonl,
    normalize_struct,
    parse_json_object_from_text,
)
from qwen4b_coco_unihuman_ft.prompts import build_box_pose_attribute_prompt  # noqa: E402


COCO_SKELETON_1_BASED = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7),
]
GT_COLOR = (40, 210, 90)
PRED_COLORS = [
    (255, 70, 70),
    (60, 170, 255),
    (255, 180, 40),
    (210, 120, 255),
    (255, 110, 190),
    (80, 230, 180),
]


def _maybe_load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --config. Install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug inference, scheme-2 eval, and visualization for box+pose+attribute VLM outputs."
    )
    parser.add_argument("--config", type=str, default="")
    parser.add_argument("--model-name-or-path", type=str, default="/home/priv/project/model/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--adapter-path", type=str, default="", help="Optional LoRA/PEFT adapter checkpoint path.")
    parser.add_argument("--jsonl", type=str, default="")
    parser.add_argument("--image-root", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="/home/priv/wmh_project/coco_person/ft_box_pose_attr_eval/debug")
    parser.add_argument("--cuda-visible-devices", type=str, default="0")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--debug-sample-size", type=int, default=10)
    parser.add_argument("--debug-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--assistant-prefix", type=str, default="")
    parser.add_argument("--extra-prompt", type=str, default="")
    parser.add_argument("--prompt-format", type=str, default="json", choices=["json"])
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--pck-thr", type=float, default=0.2)
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-gt", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-pred", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vis-dir", type=str, default="")
    parser.add_argument("--pred-jsonl", type=str, default="")
    parser.add_argument("--metrics-json", type=str, default="")
    args = parser.parse_args()

    if args.config:
        provided = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                provided.add(tok[2:].split("=")[0].replace("-", "_"))
        cfg = _maybe_load_yaml(args.config)
        for k, v in cfg.items():
            arg_name = k.replace("-", "_")
            if hasattr(args, arg_name) and arg_name not in provided:
                setattr(args, arg_name, v)
    return args


def _resolve_image_path(rec: Dict[str, Any], image_root: str) -> str:
    image = str(rec.get("image_abs", rec.get("image", "")))
    p = Path(image)
    if p.is_absolute():
        return str(p)
    if image_root:
        return str(Path(image_root) / image)
    return image


def _select_records(records: List[Dict[str, Any]], max_samples: Optional[int], debug_sample_size: int, debug_seed: int) -> List[Dict[str, Any]]:
    if max_samples is not None:
        records = records[: int(max_samples)]
    if int(debug_sample_size) > 0 and int(debug_sample_size) < len(records):
        rng = random.Random(int(debug_seed))
        indices = sorted(rng.sample(range(len(records)), k=int(debug_sample_size)))
        return [records[i] for i in indices]
    return records


def _get_dtype(args: argparse.Namespace):
    if bool(args.bf16):
        return torch.bfloat16
    if bool(args.fp16):
        return torch.float16
    return torch.float32


def _load_model(model_name_or_path: str, dtype):
    kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    has_vision = bool(getattr(cfg, "vision_config", None))

    if has_vision and Qwen3_5ForConditionalGeneration is not None:
        try:
            return Qwen3_5ForConditionalGeneration.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            pass
    if AutoModelForVision2Seq is not None:
        try:
            return AutoModelForVision2Seq.from_pretrained(model_name_or_path, **kwargs)
        except Exception:
            pass
    return AutoModelForCausalLM.from_pretrained(model_name_or_path, **kwargs)


def _model_accepts_vision_inputs(model) -> bool:
    keys = set()
    for fn_name in ("forward", "prepare_inputs_for_generation"):
        fn = getattr(model, fn_name, None)
        if fn is None:
            continue
        try:
            keys.update(inspect.signature(fn).parameters.keys())
        except Exception:
            continue
    return any(k in keys for k in ("pixel_values", "image_grid_thw", "images"))


def _filter_generate_inputs(model, inputs: Dict[str, Any]) -> Dict[str, Any]:
    keys = {"input_ids", "attention_mask", "position_ids", "token_type_ids", "past_key_values", "use_cache"}
    for fn_name in ("forward", "prepare_inputs_for_generation"):
        fn = getattr(model, fn_name, None)
        if fn is None:
            continue
        try:
            keys.update(inspect.signature(fn).parameters.keys())
        except Exception:
            continue
    return {k: v for k, v in inputs.items() if k in keys}


def _build_chat_text(processor, prompt: str, assistant_prefix: str) -> str:
    if hasattr(processor, "apply_chat_template"):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = f"<image>\n{prompt}\nAnswer:"
    if assistant_prefix:
        text += assistant_prefix
    return text


def _make_inputs(processor, texts: List[str], images: List[Image.Image], max_length: int):
    return processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=int(max_length),
    )


def _write_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _raw_box_candidate_count(obj: Any) -> int:
    if not isinstance(obj, dict):
        return 0
    for key in ("box", "boxes", "bbox", "bboxes", "detections"):
        value = obj.get(key, None)
        if isinstance(value, list) and len(value) >= 4 and all(not isinstance(x, (list, tuple, dict)) for x in value[:4]):
            return 1
        if isinstance(value, list):
            return len(value)
    persons = obj.get("persons", obj.get("person", None))
    if isinstance(persons, list):
        return len(persons)
    return 0


def _raw_tag_box_candidate_count(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    return len(re.findall(r"<box>.*?(?:</box>|$)", text, flags=re.DOTALL | re.IGNORECASE))


def _raw_tag_attr_candidate_count(text: str) -> int:
    if not isinstance(text, str) or not text:
        return 0
    genders = len(re.findall(r"<gender>.*?(?:</gender>|$)", text, flags=re.DOTALL | re.IGNORECASE))
    ages = len(re.findall(r"<age>.*?(?:</age>|$)", text, flags=re.DOTALL | re.IGNORECASE))
    return max(genders, ages)


def _draw_boxes(draw: ImageDraw.ImageDraw, boxes: List[Dict[str, Any]], color: Tuple[int, int, int], prefix: str) -> None:
    for i, item in enumerate(boxes):
        box = item.get("bbox_xyxy", [])
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{prefix}{item.get('person_id', i + 1)}"
        if "score" in item and prefix.startswith("P"):
            label += f" {float(item.get('score', 1.0)):.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color)


def _draw_pose(draw: ImageDraw.ImageDraw, poses: List[Dict[str, Any]], color: Tuple[int, int, int]) -> None:
    for pose in poses:
        kpts = pose.get("keypoints", [])
        if not isinstance(kpts, list):
            continue
        for a, b in COCO_SKELETON_1_BASED:
            ia, ib = a - 1, b - 1
            if ia >= len(kpts) or ib >= len(kpts):
                continue
            ka, kb = kpts[ia], kpts[ib]
            if not isinstance(ka, list) or not isinstance(kb, list) or len(ka) < 3 or len(kb) < 3:
                continue
            if float(ka[2]) <= 0 or float(kb[2]) <= 0:
                continue
            draw.line([(float(ka[0]), float(ka[1])), (float(kb[0]), float(kb[1]))], fill=color, width=2)
        for kp in kpts:
            if not isinstance(kp, list) or len(kp) < 3 or float(kp[2]) <= 0:
                continue
            x, y, v = float(kp[0]), float(kp[1]), float(kp[2])
            r = 3
            point_color = color if v > 1.5 else (255, 230, 40)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=point_color)


def _draw_attrs(draw: ImageDraw.ImageDraw, boxes: List[Dict[str, Any]], attrs: List[Dict[str, Any]], color: Tuple[int, int, int], y_offset: int) -> None:
    attrs_by_id = {a.get("person_id"): a for a in attrs if a.get("person_id", None) is not None}
    for i, box_item in enumerate(boxes):
        box = box_item.get("bbox_xyxy", [])
        if not isinstance(box, list) or len(box) != 4:
            continue
        attr = attrs_by_id.get(box_item.get("person_id", None))
        if attr is None and i < len(attrs):
            attr = attrs[i]
        if not isinstance(attr, dict):
            continue
        parts = []
        if "gender" in attr:
            parts.append(f"g={attr['gender']}")
        if "age" in attr:
            parts.append(f"a={float(attr['age']):.0f}")
        if parts:
            draw.text((float(box[0]) + 2, min(float(box[3]) + y_offset, 10_000)), ",".join(parts), fill=color)


def _save_visualization(
    image: Image.Image,
    gt: Dict[str, Any],
    pred: Dict[str, Any],
    out_path: str,
    draw_gt: bool,
    draw_pred: bool,
) -> None:
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    if draw_gt:
        _draw_boxes(draw, gt["box"], GT_COLOR, "G")
        _draw_pose(draw, gt["pose"], GT_COLOR)
        _draw_attrs(draw, gt["box"], gt["attribute"], GT_COLOR, 4)
    if draw_pred:
        _draw_boxes(draw, pred["box"], PRED_COLORS[0], "P")
        _draw_pose(draw, pred["pose"], PRED_COLORS[0])
        _draw_attrs(draw, pred["box"], pred["attribute"], PRED_COLORS[0], 18)
    summary = f"GT box:{len(gt['box'])}  PRED box:{len(pred['box'])}"
    draw.rectangle([4, 4, 260, 24], fill=(0, 0, 0))
    draw.text((8, 8), summary, fill=(255, 255, 255))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    vis.save(out_path)


def main() -> None:
    args = parse_args()
    if not str(args.jsonl).strip():
        raise ValueError("Missing --jsonl or yaml `jsonl`.")
    if str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = Path(args.vis_dir) if str(args.vis_dir).strip() else out_dir / "vis"
    pred_only_vis_dir = out_dir / "vis_pred_only"
    pred_jsonl = Path(args.pred_jsonl) if str(args.pred_jsonl).strip() else out_dir / "predictions.jsonl"
    metrics_json = Path(args.metrics_json) if str(args.metrics_json).strip() else out_dir / "metrics.json"

    all_records = load_jsonl(str(args.jsonl))
    records = _select_records(
        all_records,
        max_samples=args.max_samples,
        debug_sample_size=int(args.debug_sample_size),
        debug_seed=int(args.debug_seed),
    )
    if not records:
        raise ValueError(f"No records loaded from {args.jsonl}. If this is val.jsonl, check val image root first.")

    processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    model = _load_model(args.model_name_or_path, _get_dtype(args))
    if str(args.adapter_path).strip():
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter_path).strip())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    if not _model_accepts_vision_inputs(model):
        raise ValueError(
            "Loaded model does not accept image inputs. "
            f"Current model: {args.model_name_or_path}. "
            "Please use a VL model (for example Qwen2.5-VL-* or Qwen3.5-VL-*)."
        )
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    if pad_token_id is None:
        pad_token_id = eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = eos_token_id

    prompt = build_box_pose_attribute_prompt(str(args.extra_prompt).strip() or None)
    assistant_prefix = str(args.assistant_prefix)
    pred_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        pbar = tqdm(range(0, len(records), int(args.batch_size)), desc="Infer box/pose/attr", dynamic_ncols=True)
        for start in pbar:
            batch_records = records[start : start + int(args.batch_size)]
            images: List[Image.Image] = []
            texts: List[str] = []
            image_paths: List[str] = []
            for rec in batch_records:
                image_path = _resolve_image_path(rec, str(args.image_root).strip())
                image = Image.open(image_path).convert("RGB")
                image_paths.append(image_path)
                images.append(image)
                texts.append(_build_chat_text(processor, prompt, assistant_prefix))

            inputs = _make_inputs(processor, texts, images, max_length=int(args.max_length))
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)
            gen_inputs = _filter_generate_inputs(model, inputs)
            gen_ids = model.generate(
                **gen_inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
            input_len = gen_inputs["input_ids"].shape[1]
            gen_suffix = gen_ids[:, input_len:]
            suffixes = processor.batch_decode(gen_suffix, skip_special_tokens=True)

            for rec, image, image_path, suffix in zip(batch_records, images, image_paths, suffixes):
                pred_text = assistant_prefix + suffix
                image_size = image.size
                obj = parse_json_object_from_text(pred_text)
                pred_struct = normalize_struct(obj, image_size=image_size)
                parsed_from = "json" if obj is not None else "none"
                gt_struct = get_record_gt(rec, image_size=image_size)
                row = {
                    "id": str(rec.get("id", "")),
                    "image": rec.get("image", ""),
                    "image_path": image_path,
                    "image_size": [image_size[0], image_size[1]],
                    "prompt": prompt,
                    "prompt_format": str(args.prompt_format),
                    "parsed_from": parsed_from,
                    "prediction_text": pred_text,
                    "valid_json": obj is not None,
                    "valid_struct": parsed_from != "none",
                    "raw_json_box_count": _raw_box_candidate_count(obj),
                    "raw_tag_box_count": _raw_tag_box_candidate_count(pred_text),
                    "post_box_count": len(pred_struct["box"]),
                    "raw_tag_attr_count": _raw_tag_attr_candidate_count(pred_text),
                    "post_attr_count": len(pred_struct["attribute"]),
                    "prediction_struct_post": pred_struct,
                    "gt_struct": gt_struct,
                }
                pred_rows.append(row)

                if bool(args.save_vis):
                    stem = Path(image_path).stem
                    out_path = vis_dir / f"{row['id']}_{stem}.jpg"
                    _save_visualization(
                        image=image,
                        gt=gt_struct,
                        pred=pred_struct,
                        out_path=str(out_path),
                        draw_gt=bool(args.draw_gt),
                        draw_pred=bool(args.draw_pred),
                    )
                    pred_only_path = pred_only_vis_dir / f"{row['id']}_{stem}.jpg"
                    _save_visualization(
                        image=image,
                        gt=gt_struct,
                        pred=pred_struct,
                        out_path=str(pred_only_path),
                        draw_gt=False,
                        draw_pred=True,
                    )

    _write_jsonl(str(pred_jsonl), pred_rows)
    metrics = evaluate_records_scheme2(
        gt_records=records,
        pred_records=pred_rows,
        iou_thr=float(args.iou_thr),
        pck_thr=float(args.pck_thr),
    )
    metrics["model_name_or_path"] = str(args.model_name_or_path)
    metrics["adapter_path"] = str(args.adapter_path)
    metrics["jsonl"] = str(args.jsonl)
    metrics["output_dir"] = str(out_dir)
    metrics["pred_jsonl"] = str(pred_jsonl)
    metrics["vis_dir"] = str(vis_dir) if bool(args.save_vis) else ""
    metrics["pred_only_vis_dir"] = str(pred_only_vis_dir) if bool(args.save_vis) else ""
    metrics["num_debug_records"] = len(records)
    match_debug_jsonl = out_dir / "match_debug.jsonl"
    _write_jsonl(str(match_debug_jsonl), metrics.get("per_image", []))
    metrics["match_debug_jsonl"] = str(match_debug_jsonl)
    metrics_for_print = dict(metrics)
    metrics_for_print.pop("per_image", None)
    dump_json(str(metrics_json), metrics)
    print(json.dumps(metrics_for_print, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
