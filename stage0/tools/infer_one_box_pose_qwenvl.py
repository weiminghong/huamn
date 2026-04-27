import argparse
import inspect
import json
import os
import os.path as osp
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm
from transformers import AutoProcessor

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from box_task.postprocess import extract_pred_boxes, nms
from common.io_utils import dump_jsonl
from common.stage0_mm_eval import Stage0BoxPoseDataset, collate_identity
from common.target_builder import parse_json_object_from_text
from stage0.tools import eval_stage0_box_pose_qwenvl as eval_mod


COCO_SKELETON_1_BASED = [
    (16, 14), (14, 12), (17, 15), (15, 13), (12, 13),
    (6, 12), (7, 13), (6, 7), (6, 8), (7, 9),
    (8, 10), (9, 11), (2, 3), (1, 2), (1, 3),
    (2, 4), (3, 5), (4, 6), (5, 7),
]

PERSON_COLORS = [
    (255, 64, 64),
    (64, 200, 255),
    (80, 220, 120),
    (255, 180, 60),
    (220, 120, 255),
    (255, 120, 180),
]


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den <= 0 else float(num) / float(den)


def _compute_parsing_metrics_from_hist(hist: np.ndarray) -> Dict[str, Any]:
    if hist.size == 0:
        return {
            "pixel_acc": 0.0,
            "mean_acc": 0.0,
            "mIoU": 0.0,
            "num_classes": 0,
            "valid_class_count": 0,
        }
    diag = np.diag(hist).astype(np.float64)
    gt_sum = hist.sum(axis=1).astype(np.float64)
    pred_sum = hist.sum(axis=0).astype(np.float64)
    union = gt_sum + pred_sum - diag
    with np.errstate(divide="ignore", invalid="ignore"):
        class_iou = np.where(union > 0, diag / union, np.nan)
        class_acc = np.where(gt_sum > 0, diag / gt_sum, np.nan)
    return {
        "pixel_acc": float(_safe_div(diag.sum(), hist.sum())),
        "mean_acc": float(np.nanmean(class_acc)) if np.any(~np.isnan(class_acc)) else 0.0,
        "mIoU": float(np.nanmean(class_iou)) if np.any(~np.isnan(class_iou)) else 0.0,
        "num_classes": int(hist.shape[0]),
        "valid_class_count": int(np.sum(union > 0)),
    }


def _build_hist_for_pair(gt_mask: np.ndarray, pred_mask: np.ndarray, num_classes: int) -> np.ndarray:
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(f"shape mismatch: gt={gt_mask.shape}, pred={pred_mask.shape}")
    gt = gt_mask.astype(np.int64, copy=False)
    pred = pred_mask.astype(np.int64, copy=False)
    valid = (gt >= 0) & (gt < num_classes) & (pred >= 0) & (pred < num_classes)
    hist = np.zeros((num_classes, num_classes), dtype=np.int64)
    if np.any(valid):
        ids = gt[valid] * num_classes + pred[valid]
        hist += np.bincount(ids, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
    return hist


def _load_parsing_gt_mask(gt_path: str) -> Optional[np.ndarray]:
    if not osp.isfile(gt_path):
        return None
    try:
        arr = np.array(Image.open(gt_path), dtype=np.uint8)
    except Exception:
        return None
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr


def _init_parsing_model_if_needed(args: argparse.Namespace):
    if not bool(args.parsing):
        return None, None

    sapiens_root = osp.join(REPO_ROOT, "sapiens-main")
    for rel in ("seg", "engine", "cv", "pretrain"):
        p = osp.join(sapiens_root, rel)
        if osp.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    from mmengine.model import revert_sync_batchnorm  # type: ignore
    from mmseg.apis import init_model  # type: ignore

    parsing_device = str(args.parsing_device).strip()
    if not parsing_device:
        parsing_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = init_model(args.parsing_config, args.parsing_checkpoint, device=parsing_device)
    model = revert_sync_batchnorm(model)
    model.cfg.test_dataloader.batch_size = 1
    return model, parsing_device


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
    parser = argparse.ArgumentParser(
        description="Multi-image box+pose inference with the same prompt/parser pipeline as eval script."
    )
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument("--jsonl", type=str, default="", help="Unified stage0 jsonl path.")
    parser.add_argument("--image-root", type=str, default="", help="Image root for relative paths in jsonl.")
    parser.add_argument("--model-path", type=str, default="/share2/program/models/Qwen3.5-4B")
    parser.add_argument("--output-dir", type=str, default="stage0/outputs/infer_box_pose_debug")
    parser.add_argument("--cuda-visible-devices", type=str, default="3")

    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--debug-sample-size", type=int, default=0, help="If >0, run on random subset.")
    parser.add_argument("--debug-seed", type=int, default=42)

    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--repair-max-new-tokens", type=int, default=1024)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")

    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--nms-iou-thr", type=float, default=0.5)
    parser.add_argument("--max-per-img", type=int, default=100)
    parser.add_argument("--vis-score-thr", type=float, default=0.0, help="Visualization score threshold.")
    parser.add_argument("--save-vis", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vis-dir", type=str, default="", help="Default: <output-dir>/vis")
    parser.add_argument("--pred-jsonl", type=str, default="", help="Default: <output-dir>/predictions.jsonl")
    parser.add_argument("--summary-json", type=str, default="", help="Default: <output-dir>/summary.json")
    parser.add_argument("--parsing", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--parsing-gt-dir",
        type=str,
        default="/share2/program/weiminghong/coco/coco_person/parsing_sapiens/parsing_labels",
        help="Directory of parsing GT id masks named as <image_stem>.png",
    )
    parser.add_argument(
        "--parsing-config",
        type=str,
        default=osp.join(
            REPO_ROOT,
            "sapiens-main",
            "seg",
            "configs",
            "sapiens_seg",
            "goliath",
            "sapiens_1b_goliath-1024x768.py",
        ),
    )
    parser.add_argument(
        "--parsing-checkpoint",
        type=str,
        default="/share2/program/weiminghong/coco/checkpoints/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth",
    )
    parser.add_argument(
        "--parsing-device",
        type=str,
        default="",
        help="e.g. cuda:0; default follows current CUDA availability.",
    )
    parser.add_argument("--parsing-num-classes", type=int, default=28)
    parser.add_argument("--save-parsing-pred", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--parsing-pred-dir",
        type=str,
        default="",
        help="Default: <output-dir>/parsing_pred when --save-parsing-pred is true",
    )
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


def _call_build_model_inputs(
    processor,
    user_texts: List[str],
    assistant_prefixes: List[str],
    images: List[Any],
    max_length: int,
) -> Dict[str, Any]:
    fn = eval_mod._build_model_inputs
    sig = inspect.signature(fn)
    params = set(sig.parameters.keys())
    if {"user_texts", "assistant_prefixes"}.issubset(params):
        return fn(
            processor=processor,
            user_texts=user_texts,
            assistant_prefixes=assistant_prefixes,
            images=images,
            max_length=max_length,
        )

    prompts = []
    for p, prefix in zip(user_texts, assistant_prefixes):
        prompts.append(f"{p}\nOutput:\n{prefix}")
    return fn(
        processor=processor,
        prompts=prompts,
        images=images,
        max_length=max_length,
    )


def _draw_one_image(
    image,
    pred_boxes: List[Dict[str, Any]],
    pred_poses: List[Dict[str, Any]],
    vis_score_thr: float,
):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)

    for i, b in enumerate(pred_boxes):
        score = float(b.get("score", 1.0)) if isinstance(b, dict) else 1.0
        if score < float(vis_score_thr):
            continue
        xyxy = b.get("bbox_xyxy", b.get("bbox")) if isinstance(b, dict) else None
        if not isinstance(xyxy, (list, tuple)) or len(xyxy) != 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        color = PERSON_COLORS[i % len(PERSON_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 14)), f"p{i}: {score:.2f}", fill=color)

    for i, p in enumerate(pred_poses):
        color = PERSON_COLORS[i % len(PERSON_COLORS)]
        if not isinstance(p, dict):
            continue
        score = float(p.get("score", 1.0))
        if score < float(vis_score_thr):
            continue
        kpts = p.get("keypoints", [])
        if not isinstance(kpts, list):
            continue

        for a, b in COCO_SKELETON_1_BASED:
            ia, ib = a - 1, b - 1
            if ia >= len(kpts) or ib >= len(kpts):
                continue
            ka = kpts[ia]
            kb = kpts[ib]
            if not isinstance(ka, (list, tuple)) or not isinstance(kb, (list, tuple)):
                continue
            if len(ka) < 3 or len(kb) < 3:
                continue
            if float(ka[2]) <= 0 or float(kb[2]) <= 0:
                continue
            draw.line([(float(ka[0]), float(ka[1])), (float(kb[0]), float(kb[1]))], fill=color, width=2)

        for kp in kpts:
            if not isinstance(kp, (list, tuple)) or len(kp) < 3:
                continue
            x, y, v = float(kp[0]), float(kp[1]), float(kp[2])
            if v <= 0:
                continue
            r = 3
            point_color = color if v > 1.5 else (255, 255, 0)
            draw.ellipse([x - r, y - r, x + r, y + r], fill=point_color)

    return vis


def main() -> None:
    args = parse_args()
    if not str(args.jsonl).strip():
        raise ValueError("Missing --jsonl (or yaml `jsonl`).")

    if str(args.cuda_visible_devices).strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices).strip()

    dataset = Stage0BoxPoseDataset(
        jsonl_path=str(args.jsonl).strip(),
        image_root=str(args.image_root).strip() if str(args.image_root).strip() else None,
        max_samples=args.max_samples,
    )

    infer_dataset = dataset
    if int(args.debug_sample_size) > 0 and int(args.debug_sample_size) < len(dataset):
        rng = random.Random(int(args.debug_seed))
        indices = sorted(rng.sample(range(len(dataset)), k=int(args.debug_sample_size)))
        infer_dataset = torch.utils.data.Subset(dataset, indices)

    dataloader = torch.utils.data.DataLoader(
        infer_dataset,
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
    vis_dir = str(args.vis_dir).strip() if str(args.vis_dir).strip() else osp.join(args.output_dir, "vis")
    if bool(args.save_vis):
        os.makedirs(vis_dir, exist_ok=True)

    pred_jsonl = str(args.pred_jsonl).strip() if str(args.pred_jsonl).strip() else osp.join(args.output_dir, "predictions.jsonl")
    summary_json = str(args.summary_json).strip() if str(args.summary_json).strip() else osp.join(args.output_dir, "summary.json")
    parsing_pred_dir = (
        str(args.parsing_pred_dir).strip()
        if str(args.parsing_pred_dir).strip()
        else osp.join(args.output_dir, "parsing_pred")
    )
    if bool(args.parsing) and bool(args.save_parsing_pred):
        os.makedirs(parsing_pred_dir, exist_ok=True)

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    truncate_fn = getattr(eval_mod, "_truncate_to_answer", None)
    results: List[Dict[str, Any]] = []
    parsing_model, parsing_device = _init_parsing_model_if_needed(args)
    parsing_infer_fn = None
    if bool(args.parsing):
        from mmseg.apis import inference_model as _mmseg_inference_model  # type: ignore
        parsing_infer_fn = _mmseg_inference_model
    parsing_hist_global = np.zeros((int(args.parsing_num_classes), int(args.parsing_num_classes)), dtype=np.int64)
    parsing_eval_count = 0
    parsing_missing_gt = 0
    parsing_failures = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, total=len(dataloader), desc="Infer", dynamic_ncols=True)
        for batch in pbar:
            user_texts = []
            assistant_prefixes = []
            for sample in batch:
                user_texts.append(f"<img>{sample.image_path}</img>\n{eval_mod.PROMPT_BOX_POSE_TAG}")
                assistant_prefixes.append("<answer>\n<person>\n<ref>person</ref>\n<box>[")

            images = [s.image for s in batch]
            inputs = _call_build_model_inputs(
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
                pred_struct_raw = eval_mod._parse_tag_struct(full_pred_text, image_size=sample.image.size)
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
                    repaired_prediction_text = eval_mod._repair_tag_once(
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
                    repaired_obj = eval_mod._parse_tag_struct(repaired_prediction_text or "", image_size=sample.image.size)
                    if repaired_obj is None:
                        repaired_obj = parse_json_object_from_text(repaired_prediction_text or "")
                    if repaired_obj is not None:
                        pred_struct_raw = repaired_obj
                        parsed_from = "repair"
                        repaired_valid = True
                        valid_struct = True

                if pred_struct_raw is None or not isinstance(pred_struct_raw, dict):
                    pred_struct_raw = {"box": [], "pose": []}
                else:
                    pred_struct_raw = eval_mod._align_box_pose(pred_struct_raw)

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
                pred_poses = eval_mod._normalize_pose_list(pred_struct_raw.get("pose", []), image_size=(w, h))

                out = {
                    "id": sample.sid,
                    "image_path": sample.image_path,
                    "image_size": [w, h],
                    "parsing_enabled": bool(args.parsing),
                    "parsed_from": parsed_from,
                    "valid_struct": bool(valid_struct),
                    "repaired_valid": bool(repaired_valid),
                    "prediction_text": full_pred_text,
                    "repaired_prediction_text": repaired_prediction_text,
                    "prediction_struct_raw": pred_struct_raw,
                    "prediction_struct_post": {
                        "box": pred_boxes,
                        "pose": pred_poses,
                    },
                }

                if bool(args.parsing):
                    parsing_obj: Dict[str, Any] = {
                        "enabled": True,
                        "gt_path": "",
                        "has_gt": False,
                        "valid": False,
                        "error": "",
                    }
                    try:
                        seg_result = parsing_infer_fn(parsing_model, str(sample.image_path))
                        pred_seg = seg_result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                        if bool(args.save_parsing_pred):
                            stem = osp.splitext(osp.basename(sample.image_path))[0]
                            pred_path = osp.join(parsing_pred_dir, f"{sample.sid}_{stem}.png")
                            Image.fromarray(pred_seg).save(pred_path)
                            parsing_obj["pred_path"] = pred_path

                        stem = osp.splitext(osp.basename(sample.image_path))[0]
                        gt_path = osp.join(str(args.parsing_gt_dir), f"{stem}.png")
                        parsing_obj["gt_path"] = gt_path
                        gt_seg = _load_parsing_gt_mask(gt_path)
                        if gt_seg is None:
                            parsing_obj["error"] = "missing_or_invalid_gt_mask"
                            parsing_missing_gt += 1
                        else:
                            hist = _build_hist_for_pair(
                                gt_mask=gt_seg,
                                pred_mask=pred_seg,
                                num_classes=int(args.parsing_num_classes),
                            )
                            metrics_one = _compute_parsing_metrics_from_hist(hist)
                            parsing_hist_global += hist
                            parsing_eval_count += 1
                            parsing_obj.update(
                                {
                                    "has_gt": True,
                                    "valid": True,
                                    "metrics": metrics_one,
                                }
                            )
                    except Exception as e:
                        parsing_obj["error"] = f"parsing_inference_exception: {str(e)}"
                        parsing_failures += 1
                    out["parsing_eval"] = parsing_obj
                else:
                    out["parsing_eval"] = {"enabled": False}

                results.append(out)

                if bool(args.save_vis):
                    vis_img = _draw_one_image(
                        image=sample.image,
                        pred_boxes=pred_boxes,
                        pred_poses=pred_poses,
                        vis_score_thr=float(args.vis_score_thr),
                    )
                    stem = osp.splitext(osp.basename(sample.image_path))[0]
                    vis_name = f"{sample.sid}_{stem}.jpg"
                    vis_img.save(osp.join(vis_dir, vis_name))

    dump_jsonl(pred_jsonl, results)
    summary = {
        "num_images": len(results),
        "parsing_enabled": bool(args.parsing),
        "num_valid_struct": sum(1 for x in results if x.get("valid_struct", False)),
        "valid_struct_rate": sum(1 for x in results if x.get("valid_struct", False)) / max(1, len(results)),
        "avg_pred_box_per_image": sum(
            len((x.get("prediction_struct_post", {}) or {}).get("box", [])) for x in results
        ) / max(1, len(results)),
        "avg_pred_pose_per_image": sum(
            len((x.get("prediction_struct_post", {}) or {}).get("pose", [])) for x in results
        ) / max(1, len(results)),
        "pred_jsonl": pred_jsonl,
        "vis_dir": vis_dir if bool(args.save_vis) else "",
    }
    if bool(args.parsing):
        summary["parsing_metrics"] = _compute_parsing_metrics_from_hist(parsing_hist_global)
        summary["parsing_metrics"].update(
            {
                "num_eval_images": int(parsing_eval_count),
                "num_missing_gt": int(parsing_missing_gt),
                "num_failures": int(parsing_failures),
                "gt_dir": str(args.parsing_gt_dir),
                "parsing_device": parsing_device,
            }
        )
        if bool(args.save_parsing_pred):
            summary["parsing_pred_dir"] = parsing_pred_dir
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
