import argparse
import json
import os
import os.path as osp
import random
import re
import sys
from typing import Any, Dict, List, Optional

import torch
from PIL import ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from box_task.postprocess import extract_pred_boxes, nms
from common.io_utils import dump_jsonl
from common.stage0_mm_eval import (
    Stage0BoxPoseDataset,
    add_sapiens_python_paths,
    collate_identity,
    evaluate_coco_official_from_predictions,
)
from common.target_builder import parse_json_object_from_text
from stage0.tools import eval_stage0_box_pose_qwenvl as eval_mod


PROMPT_BOX_ONLY_TAG = (
    'Detect every person in the image.\n'
    'Output only XML blocks wrapped inside <answer>...</answer>.\n\n'
    'Rules:\n'
    '- Coordinates are normalized to [0,1000].\n'
    '- Output one <person> block for each detected person.\n'
    '- Each block must contain exactly one <box>[x1, y1, x2, y2]</box>.\n'
    '- Output XML only. No explanation.\n\n'
    '--- EXAMPLE OUTPUT ---\n'
    '<answer>\n'
    '<person>\n'
    '<box>[120, 250, 480, 820]</box>\n'
    '<score>0.98</score>\n'
    '</person>\n'
    '<person>\n'
    '<box>[520, 200, 850, 900]</box>\n'
    '<score>0.91</score>\n'
    '</person>\n'
    '</answer>\n'
    '--- END OF EXAMPLE ---\n'
)

PROMPT_REPAIR_BOX_ONLY = (
    "Rewrite raw output into strict XML only.\n"
    "Wrap all output in <answer>...</answer>.\n"
    "Rules:\n"
    "1. Coordinates MUST be normalized to [0, 1000].\n"
    "2. Use one <person> block per detection.\n"
    "3. Each <person> block must contain one <box> and one <score>.\n"
    "--- RAW TEXT START ---\n"
)

MODEL_PATH_4B = "/home/priv/project/model/Qwen3.5-4B"
MODEL_PATH_35B_A3B = "/home/priv/project/model/Qwen3.5-35B-A3B"


def _maybe_load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError("PyYAML is required for --config. Please install with: pip install pyyaml") from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen models on box-only detection with COCO bbox metrics.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--jsonl", type=str, default="", help="Stage0 eval jsonl path. Leave empty and fill later.")
    parser.add_argument("--image-root", type=str, default="", help="Image root. Leave empty and fill later.")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH_4B)
    parser.add_argument("--output-dir", type=str, default="qwenval/outputs/box_eval")
    parser.add_argument("--cuda-visible-devices", type=str, default="0")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug-sample-size", type=int, default=0)
    parser.add_argument("--debug-seed", type=int, default=42)
    parser.add_argument("--debug-print", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--debug-save-file",
        type=str,
        default="",
        help="Optional json file to save random-sample inference details.",
    )

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--repair-max-new-tokens", type=int, default=512)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--nms-iou-thr", type=float, default=0.5)
    parser.add_argument("--max-per-img", type=int, default=100)

    parser.add_argument("--coco-ann-json", type=str, default="", help="COCO-like ann json path. Leave empty and fill later.")
    parser.add_argument("--coco-split-txt", type=str, default="", help="Split txt path. Leave empty and fill later.")
    parser.add_argument("--debug-dump-file", type=str, default="")
    parser.add_argument("--skip-coco-eval", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--vis-dir", type=str, default="", help="Default: <output-dir>/vis")
    parser.add_argument("--vis-score-thr", type=float, default=0.0)
    parser.add_argument("--draw-gt-box", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.config:
        provided = set()
        for tok in sys.argv[1:]:
            if tok.startswith("--"):
                provided.add(tok[2:].split("=")[0].replace("-", "_"))
        cfg = _maybe_load_yaml(args.config)
        for k, v in cfg.items():
            arg_name = k.replace("-", "_")
            if not hasattr(args, arg_name):
                continue
            if arg_name in provided:
                continue
            setattr(args, arg_name, v)
    return args


def _repair_box_once(
    model,
    processor,
    device: torch.device,
    raw_text: str,
    max_length: int,
    repair_max_new_tokens: int,
    pad_token_id: int,
) -> str:
    user_text = PROMPT_REPAIR_BOX_ONLY + raw_text + "\n--- RAW TEXT END ---\n"
    assistant_prefix = "<answer>\n<person>\n<box>["
    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
        txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        txt = txt + assistant_prefix
    else:
        txt = f"{user_text}\nOutput:\n{assistant_prefix}"

    tok = processor.tokenizer(
        [txt],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    for k, v in tok.items():
        if isinstance(v, torch.Tensor):
            tok[k] = v.to(device)

    gen_ids = model.generate(
        **tok,
        max_new_tokens=int(repair_max_new_tokens),
        do_sample=False,
        num_beams=1,
        pad_token_id=pad_token_id,
    )
    input_len = tok["input_ids"].shape[1]
    gen_suffix = gen_ids[:, input_len:]
    decoded = processor.batch_decode(gen_suffix, skip_special_tokens=True)[0]
    return assistant_prefix + decoded


def _parse_box_only_struct(text: str, image_size) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if "No person detected" in text:
        return {"box": [], "pose": []}

    width, height = image_size
    person_blobs = re.split(r"<person>", text, flags=re.IGNORECASE)[1:]
    if not person_blobs:
        return None

    boxes = []
    num_pattern = r"-?(?:\d+\.\d+|\d+|\.\d+)"
    for blob in person_blobs:
        score = 1.0
        m_score = re.search(r"<score>\s*([0-9]*\.?[0-9]+)", blob, flags=re.IGNORECASE)
        if m_score:
            try:
                score = float(m_score.group(1))
            except Exception:
                score = 1.0
        score = max(0.0, min(score, 1.0))

        m_box = re.search(r"<box>(.*?)(?:</box>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        if m_box is None:
            continue
        box_nums = re.findall(num_pattern, m_box.group(1))
        if len(box_nums) < 4:
            continue
        try:
            x1_norm, y1_norm, x2_norm, y2_norm = [float(n) for n in box_nums[:4]]
        except Exception:
            continue

        x1 = max(0.0, min((x1_norm / 1000.0) * width, width - 1))
        y1 = max(0.0, min((y1_norm / 1000.0) * height, height - 1))
        x2 = max(0.0, min((x2_norm / 1000.0) * width, width - 1))
        y2 = max(0.0, min((y2_norm / 1000.0) * height, height - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append({"bbox": [x1, y1, x2, y2], "score": score})

    if not boxes:
        return None
    return {"box": boxes, "pose": []}


def _build_eval_dataset(
    jsonl_path: str,
    image_root: str,
    max_samples: Optional[int],
    debug_sample_size: int,
    debug_seed: int,
):
    dataset = Stage0BoxPoseDataset(
        jsonl_path=jsonl_path,
        image_root=image_root if image_root else None,
        max_samples=max_samples,
    )
    eval_dataset = dataset
    sampled_indices: List[int] = []
    if int(debug_sample_size) > 0 and int(debug_sample_size) < len(dataset):
        rng = random.Random(int(debug_seed))
        sampled_indices = sorted(rng.sample(range(len(dataset)), k=int(debug_sample_size)))
        eval_dataset = torch.utils.data.Subset(dataset, sampled_indices)
    return eval_dataset, sampled_indices


def _draw_box_vis(
    image,
    gt_boxes: List[Any],
    pred_boxes: List[Dict[str, Any]],
    vis_score_thr: float,
    draw_gt_box: bool,
):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    if bool(draw_gt_box):
        for g in gt_boxes if isinstance(gt_boxes, list) else []:
            if isinstance(g, dict):
                xyxy = g.get("bbox_xyxy", g.get("bbox"))
            else:
                xyxy = g
            if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
                continue
            try:
                x1, y1, x2, y2 = [float(v) for v in xyxy]
            except Exception:
                continue
            if x2 <= x1 or y2 <= y1:
                continue
            draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)

    for i, p in enumerate(pred_boxes if isinstance(pred_boxes, list) else []):
        if isinstance(p, dict):
            xyxy = p.get("bbox_xyxy", p.get("bbox"))
            score = float(p.get("score", 1.0))
        else:
            xyxy = p
            score = 1.0
        if score < float(vis_score_thr):
            continue
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in xyxy]
        except Exception:
            continue
        if x2 <= x1 or y2 <= y1:
            continue
        color = (255, 80, 80) if i % 2 == 0 else (255, 160, 80)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"pred:{score:.2f}", fill=color)
    return vis


def _infer_and_collect_pred_lines(
    model,
    processor,
    device: torch.device,
    dataloader,
    args: argparse.Namespace,
):
    pred_lines: List[Dict[str, Any]] = []
    debug_lines: List[Dict[str, Any]] = []

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    truncate_fn = getattr(eval_mod, "_truncate_to_answer", None)

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc="Qwen Box Eval", dynamic_ncols=True)
        for batch in pbar:
            user_texts = []
            assistant_prefixes = []
            for sample in batch:
                user_texts.append(f"<img>{sample.image_path}</img>\n{PROMPT_BOX_ONLY_TAG}")
                assistant_prefixes.append("<answer>\n<person>\n<box>[")

            images = [s.image for s in batch]
            inputs = eval_mod._build_model_inputs(
                processor=processor,
                user_texts=user_texts,
                assistant_prefixes=assistant_prefixes,
                images=images,
                max_length=int(args.max_length),
            )
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(device)

            gen_ids = model.generate(
                **inputs,
                max_new_tokens=int(args.max_new_tokens),
                do_sample=False,
                num_beams=1,
                pad_token_id=pad_token_id,
            )
            input_len = inputs["input_ids"].shape[1]
            gen_suffix = gen_ids[:, input_len:]
            pred_suffixes = processor.batch_decode(gen_suffix, skip_special_tokens=True)

            for sample, pred_suffix, prefix in zip(batch, pred_suffixes, assistant_prefixes):
                full_pred_text = prefix + pred_suffix
                if callable(truncate_fn):
                    full_pred_text = truncate_fn(full_pred_text)

                parsed_from = "none"
                pred_struct_raw = _parse_box_only_struct(full_pred_text, image_size=sample.image.size)
                if pred_struct_raw is not None:
                    parsed_from = "tag"
                else:
                    pred_struct_raw = parse_json_object_from_text(full_pred_text)
                    if pred_struct_raw is not None:
                        parsed_from = "json"

                valid_struct = pred_struct_raw is not None
                repaired_prediction_text = None
                repaired_valid = False

                if not valid_struct:
                    repaired_prediction_text = _repair_box_once(
                        model=model,
                        processor=processor,
                        device=device,
                        raw_text=full_pred_text,
                        max_length=int(args.max_length),
                        repair_max_new_tokens=int(args.repair_max_new_tokens),
                        pad_token_id=pad_token_id,
                    )
                    if callable(truncate_fn):
                        repaired_prediction_text = truncate_fn(repaired_prediction_text or "")
                    repaired_obj = _parse_box_only_struct(repaired_prediction_text or "", image_size=sample.image.size)
                    if repaired_obj is None:
                        repaired_obj = parse_json_object_from_text(repaired_prediction_text or "")
                    if repaired_obj is not None:
                        pred_struct_raw = repaired_obj
                        parsed_from = "repair"
                        repaired_valid = True
                        valid_struct = True

                if pred_struct_raw is None or not isinstance(pred_struct_raw, dict):
                    pred_struct_raw = {"box": [], "pose": []}

                w, h = sample.image.size
                pred_boxes = extract_pred_boxes(
                    pred_struct=pred_struct_raw,
                    score_thr=float(args.score_thr),
                    image_size=[w, h],
                )
                pred_boxes = nms(
                    boxes=pred_boxes,
                    iou_thr=float(args.nms_iou_thr),
                    max_per_img=int(args.max_per_img),
                )
                pred_struct_post = {"box": pred_boxes, "pose": []}

                pred_rec = {
                    "id": sample.sid,
                    "image_path": sample.image_path,
                    "prompt": PROMPT_BOX_ONLY_TAG,
                    "prediction_text": full_pred_text,
                    "repaired_prediction_text": repaired_prediction_text,
                    "parsed_from": parsed_from,
                    "prediction_struct_raw": pred_struct_raw,
                    "prediction_struct_post": pred_struct_post,
                    "gt_struct": {"box": sample.gt_box, "pose": []},
                    "valid_struct": valid_struct,
                    "repaired_valid": repaired_valid,
                    "valid_json": valid_struct,
                }
                pred_lines.append(pred_rec)

                debug_rec = {
                    "id": sample.sid,
                    "image_path": sample.image_path,
                    "parsed_from": parsed_from,
                    "valid_struct": valid_struct,
                    "gt_num_box": len(sample.gt_box) if isinstance(sample.gt_box, list) else 0,
                    "pred_num_box": len(pred_boxes),
                    "gt_box": sample.gt_box,
                    "pred_box_post": pred_boxes,
                    "prediction_text": full_pred_text,
                    "repaired_prediction_text": repaired_prediction_text,
                }
                debug_lines.append(debug_rec)

                if bool(args.debug_print):
                    print(
                        json.dumps(
                            {
                                "id": sample.sid,
                                "image_path": sample.image_path,
                                "gt_num_box": debug_rec["gt_num_box"],
                                "pred_num_box": debug_rec["pred_num_box"],
                                "parsed_from": parsed_from,
                            },
                            ensure_ascii=False,
                        )
                    )

                if bool(args.save_vis):
                    vis_img = _draw_box_vis(
                        image=sample.image,
                        gt_boxes=sample.gt_box,
                        pred_boxes=pred_boxes,
                        vis_score_thr=float(args.vis_score_thr),
                        draw_gt_box=bool(args.draw_gt_box),
                    )
                    stem = osp.splitext(osp.basename(sample.image_path))[0]
                    vis_name = f"{sample.sid}_{stem}.jpg"
                    vis_img.save(osp.join(args.vis_dir, vis_name))
    return pred_lines, debug_lines


def main() -> None:
    args = parse_args()
    required = {
        "jsonl": str(args.jsonl).strip(),
        "image_root": str(args.image_root).strip(),
        "coco_ann_json": str(args.coco_ann_json).strip(),
        "coco_split_txt": str(args.coco_split_txt).strip(),
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        raise ValueError(
            f"Missing required paths: {missing}. "
            "This script is intentionally created with empty placeholders for the new server."
        )

    if str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()

    add_sapiens_python_paths(REPO_ROOT)

    eval_dataset, sampled_indices = _build_eval_dataset(
        jsonl_path=args.jsonl,
        image_root=args.image_root,
        max_samples=args.max_samples,
        debug_sample_size=int(args.debug_sample_size),
        debug_seed=int(args.debug_seed),
    )

    dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_identity,
    )

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    model = eval_mod.build_model(args.model_path, eval_mod.get_torch_dtype(args.fp16, args.bf16))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)
    if not str(args.vis_dir).strip():
        args.vis_dir = osp.join(args.output_dir, "vis")
    if bool(args.save_vis):
        os.makedirs(args.vis_dir, exist_ok=True)
    pred_lines, debug_lines = _infer_and_collect_pred_lines(
        model=model,
        processor=processor,
        device=device,
        dataloader=dataloader,
        args=args,
    )

    metrics = {
        "task": "box_only",
        "model_path": str(args.model_path),
        "num_samples": len(pred_lines),
        "valid_struct_rate": sum(1 for x in pred_lines if x.get("valid_struct", False)) / max(1, len(pred_lines)),
        "save_vis": bool(args.save_vis),
        "vis_dir": str(args.vis_dir) if bool(args.save_vis) else "",
    }
    if not bool(args.skip_coco_eval):
        coco_metrics = evaluate_coco_official_from_predictions(
            pred_lines=pred_lines,
            ann_json=str(args.coco_ann_json).strip(),
            split_txt=str(args.coco_split_txt).strip() if str(args.coco_split_txt).strip() else None,
        )
        metrics.update(coco_metrics)
    else:
        metrics["coco_eval_skipped"] = True

    dump_jsonl(osp.join(args.output_dir, "predictions.jsonl"), pred_lines)
    debug_dump_file = str(args.debug_dump_file).strip()
    if not debug_dump_file:
        debug_dump_file = osp.join(args.output_dir, "debug_samples.jsonl")
    dump_jsonl(debug_dump_file, debug_lines)
    debug_save_file = str(args.debug_save_file).strip()
    if int(args.debug_sample_size) > 0:
        if not debug_save_file:
            debug_save_file = osp.join(args.output_dir, "random_infer_results.json")
        random_payload = {
            "mode": "random_debug",
            "debug_sample_size": int(args.debug_sample_size),
            "debug_seed": int(args.debug_seed),
            "sampled_indices": sampled_indices,
            "num_samples": len(debug_lines),
            "samples": debug_lines,
        }
        with open(debug_save_file, "w", encoding="utf-8") as f:
            json.dump(random_payload, f, ensure_ascii=False, indent=2)
        metrics["debug_save_file"] = debug_save_file

    with open(osp.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
