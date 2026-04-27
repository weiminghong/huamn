import argparse
import json
import os
import os.path as osp
import random
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor

try:
    from transformers import AutoModelForVision2Seq
except Exception:
    AutoModelForVision2Seq = None

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
except Exception:
    Qwen3_5ForConditionalGeneration = None

REPO_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
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

# [核心修改 1]：引入真实具体的 Few-Shot 示例，消除模型复读机行为
# PROMPT_BOX_POSE_TAG = (
#     'Detect every person in the image.\n'
#     'Output only XML blocks wrapped inside <answer>...</answer>.\n\n'
#     'Rules:\n'
#     '- Coordinates are normalized to [0,1000].\n'
#     '- Pose contains exactly 17 keypoints in COCO order:\n'
#     '  nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.\n'
#     '- Visibility: v=2 visible, v=1 occluded/inferred.\n'
#     '- If a keypoint is missing or out-of-frame, use [0,0,0].\n'
#     '- Output XML only. No explanation.\n\n'
#     '--- EXAMPLE OUTPUT ---\n'
#     '<answer>\n'
#     '<person>\n'
#     '<box>[150, 200, 450, 800]</box>\n'
#     # '<pose>[250, 300, 2], [260, 280, 2], [240, 280, 2], [280, 290, 2], [220, 290, 2], [300, 350, 2], [200, 350, 2], [350, 450, 2], [150, 450, 2], [380, 500, 2], [120, 500, 2], [280, 600, 2], [220, 600, 2], [290, 750, 2], [210, 750, 2], [300, 880, 2], [200, 880, 2]</pose>\n'
#    # '<pose>[320, 280, 2], [340, 260, 2], [290, 270, 2], [370, 290, 2], [270, 300, 1], [390, 380, 2], [250, 400, 2], [450, 460, 2], [210, 510, 2], [410, 550, 2], [230, 600, 2], [360, 620, 2], [280, 630, 2], [420, 760, 2], [250, 790, 2], [380, 890, 2], [290, 930, 2]</pose>\n'
#    '<pose>[461, 110, 2], [467, 106, 2], [454, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2], [467, 106, 2], [461, 106, 2]</pose>\n'
#     '<score>0.98</score>\n'
#     '</person>\n'
#     '</answer>\n'
#     '--- END OF EXAMPLE ---\n'
# )
PROMPT_BOX_POSE_TAG = (
    'Detect every person in the image.\n'
    'Output only XML blocks wrapped inside <answer>...</answer>.\n\n'
    'Rules:\n'
    '- Coordinates are normalized to [0,1000].\n'
    '- Pose contains exactly 17 keypoints in COCO order:\n'
    '  nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle.\n'
    '- Format: Each keypoint is a triplet [x, y, v].\n'
    '  * x, y: coordinates.\n'
    '  * v (visibility): 2 = visible, 1 = occluded/inferred, 0 = missing/out-of-frame.\n'
    '- If a keypoint is missing (v=0), you MUST output exactly [0,0,0].\n'
    '- Output XML only. No explanation.\n\n'
    '--- EXAMPLE OUTPUT ---\n'
    '<answer>\n'
    '<person>\n'
    '<box>[120, 250, 480, 820]</box>\n'
    '<pose>[250, 300, 2], [260, 280, 2], [240, 280, 2], [280, 290, 2], [220, 290, 2], [300, 350, 2], [200, 350, 2], [350, 450, 2], [150, 450, 2], [380, 500, 1], [120, 500, 2], [280, 600, 2], [220, 600, 2], [290, 750, 1], [210, 750, 1], [0, 0, 0], [0, 0, 0]</pose>\n'
    '<score>0.98</score>\n'
    '</person>\n'
    '</answer>\n'
    '--- END OF EXAMPLE ---\n'
)

# 修复用的 Prompt 也一并精简：
PROMPT_REPAIR_TAG = (
    "Rewrite raw output into strict XML only.\n"
    "Wrap all output in <answer>...</answer>.\n"
    "Rules:\n"
    "1. Coordinates MUST be normalized to [0, 1000].\n"
    "2. Pose MUST contain exactly 17 slots in COCO order. If missing, use [0,0,0].\n"
    "--- RAW TEXT START ---\n"
)

# PROMPT_REPAIR_TAG = (
#     "Rewrite raw output into strict XML only.\n"
#     "Wrap all output in <answer>...</answer>.\n"
#     "Rules:\n"
#     "1. ALL coordinates MUST be normalized in the range [0, 1000].\n"
#     "2. Pose MUST contain exactly 17 slots in COCO order. If missing, use [0,0,0].\n"
#     "--- RAW TEXT START ---\n"
# )

PROMPT_BOX_POSE = PROMPT_BOX_POSE_TAG


def _maybe_load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "PyYAML is required for --config. Please install with: pip install pyyaml"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Stage0 eval for box+pose with Qwen-VL.')
    parser.add_argument('--config', type=str, default=None, help='Optional YAML config path.')
    parser.add_argument('--jsonl', type=str, default='', help='Unified jsonl path.')
    parser.add_argument('--image-root', type=str, default='', help='Optional image root for relative image paths.')
    parser.add_argument('--model-path', type=str, default='/share2/program/models/Qwen3.5-4B')
    parser.add_argument('--output-dir', type=str, default='stage0/outputs/box_pose_eval')
    parser.add_argument('--cuda-visible-devices', type=str, default='', help='e.g. "3"')

    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=2)

    parser.add_argument('--max-length', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=1536)
    parser.add_argument('--repair-max-new-tokens', type=int, default=1024)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')

    parser.add_argument('--score-thr', type=float, default=0.0)
    parser.add_argument('--nms-iou-thr', type=float, default=0.5)
    parser.add_argument('--max-per-img', type=int, default=100)

    parser.add_argument('--eval-iou-thr', type=float, default=0.5)
    parser.add_argument('--map-nproc', type=int, default=1)
    parser.add_argument('--coco-ann-json', type=str, default='',
                        help='person_with_instances.json for official COCO eval.')
    parser.add_argument('--coco-split-txt', type=str, default='',
                        help='Optional split txt (e.g., val.txt) for official COCO eval.')
    parser.add_argument('--debug-sample-size', type=int, default=0,
                        help='If >0, evaluate a random subset of this size.')
    parser.add_argument('--debug-seed', type=int, default=42, help='Random seed for debug subset sampling.')
    parser.add_argument('--debug-dump-file', type=str, default='', help='Optional per-image debug jsonl output path.')
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


def get_torch_dtype(fp16: bool, bf16: bool):
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return torch.float32


def build_model(model_path: str, torch_dtype):
    if Qwen3_5ForConditionalGeneration is not None:
        try:
            return Qwen3_5ForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
        except Exception:
            pass

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


def _zero_pose_17() -> List[List[float]]:
    return [[0.0, 0.0, 0.0] for _ in range(17)]


def _align_box_pose(struct_obj: Dict[str, Any]) -> Dict[str, Any]:
    boxes = struct_obj.get("box", [])
    poses = struct_obj.get("pose", [])
    if not isinstance(boxes, list):
        boxes = []
    if not isinstance(poses, list):
        poses = []

    aligned_poses: List[Dict[str, Any]] = []
    for i, box_item in enumerate(boxes):
        score = 1.0
        if isinstance(box_item, dict):
            try:
                score = float(box_item.get("score", 1.0))
            except Exception:
                score = 1.0

        if i < len(poses) and isinstance(poses[i], dict):
            kpts = poses[i].get("keypoints", [])
            if isinstance(kpts, list) and len(kpts) == 17:
                aligned_poses.append({"keypoints": kpts, "score": score})
                continue
        aligned_poses.append({"keypoints": _zero_pose_17(), "score": score})
    return {"box": boxes, "pose": aligned_poses}


# [核心修改 2]：对 repair 同样执行严格的 Chat Template Injection
def _repair_tag_once(
        model,
        processor,
        device: torch.device,
        raw_text: str,
        max_length: int,
        repair_max_new_tokens: int,
        pad_token_id: int,
) -> str:
    user_text = PROMPT_REPAIR_TAG + raw_text + "\n--- RAW TEXT END ---\n"
    assistant_prefix = "<answer>\n<person>\n<ref>person</ref>\n<box>["

    if hasattr(processor, "apply_chat_template"):
        messages = [{"role": "user", "content": [{"type": "text", "text": user_text}]}]
        txt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        txt = txt + assistant_prefix
    else:
        txt = f"{user_text}\nOutput:\n{assistant_prefix}"

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
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
            max_new_tokens=repair_max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
        )
        input_len = tok["input_ids"].shape[1]
        gen_suffix = gen_ids[:, input_len:]
        decoded = processor.batch_decode(gen_suffix, skip_special_tokens=True)[0]
        # 必须将前缀拼接回去返回
        return assistant_prefix + decoded
    return ""


def _truncate_to_answer(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""
    start_m = re.search(r"<answer>", text, flags=re.IGNORECASE)
    if start_m:
        text = text[start_m.start():]
    end_m = re.search(r"</answer>", text, flags=re.IGNORECASE)
    if end_m:
        text = text[: end_m.end()]
    return text


# [核心修改 3]：彻底分离 user_texts 和 assistant_prefixes，确保物理堵嘴突破 User 回合封锁
def _build_model_inputs(processor, user_texts: List[str], assistant_prefixes: List[str], images: List[Any],
                        max_length: int):
    texts: List[str] = []
    if hasattr(processor, "apply_chat_template"):
        for p, prefix in zip(user_texts, assistant_prefixes):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": p},
                    ],
                }
            ]
            # 生成带 <|im_start|>assistant 结尾的模板
            txt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            # 在 assistant 身份下强行续接内容
            txt = txt + prefix
            texts.append(txt)
    else:
        for p, prefix in zip(user_texts, assistant_prefixes):
            texts.append(f"{p}\nOutput:\n{prefix}")

    return processor(
        text=texts,
        images=images,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=max_length,
    )


def _to_image_xy(v: float, length: int) -> float:
    if length <= 1:
        return 0.0
    if 0.0 <= v <= 1.0:
        out = v * float(length - 1)
    elif 0.0 <= v <= 1000.0:
        out = (v / 1000.0) * float(length - 1)
    else:
        out = v
    return max(0.0, min(out, float(length - 1)))


def _normalize_pose_list(poses: Any, image_size: Tuple[int, int]) -> List[Dict[str, Any]]:
    width, height = image_size
    if not isinstance(poses, list):
        return []

    out = []
    for item in poses:
        if isinstance(item, dict):
            kpts = item.get('keypoints', [])
            score = item.get('score', 1.0)
        elif isinstance(item, list):
            kpts = item
            score = 1.0
        else:
            continue

        if not isinstance(kpts, list):
            continue

        nk = []
        for kp in kpts:
            if not isinstance(kp, (list, tuple)) or len(kp) < 2:
                continue
            try:
                x = float(kp[0])
                y = float(kp[1])
                v = float(kp[2]) if len(kp) > 2 else 2.0
            except Exception:
                continue
            x = max(0.0, min(x, width - 1))
            y = max(0.0, min(y, height - 1))
            nk.append([x, y, v])

        if len(nk) == 0:
            continue
        if len(nk) < 17:
            nk.extend([[0.0, 0.0, 0.0] for _ in range(17 - len(nk))])
        nk = nk[:17]
        try:
            score = float(score)
        except Exception:
            score = 1.0
        out.append({'keypoints': nk, 'score': score})
    return out


def _xyxy_to_xywh(box: List[float]) -> List[float]:
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _pose_to_coco_keypoints_51(pose_obj: Any) -> List[float]:
    if isinstance(pose_obj, dict):
        kpts = pose_obj.get('keypoints', [])
    else:
        kpts = pose_obj
    out: List[float] = []
    if not isinstance(kpts, list):
        kpts = []
    for i in range(17):
        if i < len(kpts) and isinstance(kpts[i], (list, tuple)) and len(kpts[i]) >= 2:
            x = float(kpts[i][0])
            y = float(kpts[i][1])
            v = float(kpts[i][2]) if len(kpts[i]) > 2 else 2.0
            if v < 0.5:
                vv = 0.0
            elif v < 1.5:
                vv = 1.0
            else:
                vv = 2.0
            out.extend([x, y, vv])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def _build_prediction_coco_instances(sample_id: str, pred_struct_post: Dict[str, Any]) -> List[Dict[str, Any]]:
    try:
        image_id = int(sample_id)
    except Exception:
        return []

    boxes = pred_struct_post.get('box', [])
    poses = pred_struct_post.get('pose', [])
    if not isinstance(boxes, list):
        boxes = []
    if not isinstance(poses, list):
        poses = []

    n = max(len(boxes), len(poses))
    results: List[Dict[str, Any]] = []
    for i in range(n):
        score_box = 1.0
        score_pose = 1.0
        bbox_xywh = [0.0, 0.0, 0.0, 0.0]
        keypoints_51 = [0.0] * 51

        if i < len(boxes) and isinstance(boxes[i], dict):
            xyxy = boxes[i].get('bbox_xyxy', boxes[i].get('bbox'))
            if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
                bbox_xywh = _xyxy_to_xywh([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])])
            try:
                score_box = float(boxes[i].get('score', 1.0))
            except Exception:
                score_box = 1.0
        if i < len(poses):
            keypoints_51 = _pose_to_coco_keypoints_51(poses[i])
            if isinstance(poses[i], dict):
                try:
                    score_pose = float(poses[i].get('score', 1.0))
                except Exception:
                    score_pose = 1.0
        score = max(0.0, min((score_box + score_pose) * 0.5, 1.0))
        results.append(
            {
                'image_id': image_id,
                'category_id': 1,
                'bbox': bbox_xywh,
                'keypoints': keypoints_51,
                'score': score,
            }
        )
    return results


def _parse_tag_struct(text: str, image_size: Tuple[int, int]) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    if "No person detected" in text:
        return {'box': [], 'pose': []}

    width, height = image_size
    person_blobs = re.split(r"<person>", text, flags=re.IGNORECASE)[1:]
    if not person_blobs:
        return None

    boxes = []
    poses = []
    num_pattern = r"-?(?:\d+\.\d+|\d+|\.\d+)"
    for blob in person_blobs:
        score = 1.0
        m_score = re.search(r"<score>\s*([0-9]*\.?[0-9]+)", blob, flags=re.IGNORECASE)
        if m_score:
            try:
                score = float(m_score.group(1))
            except Exception:
                pass
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

        m_pose = re.search(r"<pose>(.*?)(?:</pose>|$)", blob, flags=re.DOTALL | re.IGNORECASE)
        kpts = []
        if m_pose is not None:
            pose_nums = re.findall(num_pattern, m_pose.group(1))

            if len(pose_nums) >= 51:
                for i in range(17):
                    try:
                        x_n = float(pose_nums[i * 3])
                        y_n = float(pose_nums[i * 3 + 1])
                        v_n = int(round(float(pose_nums[i * 3 + 2])))
                    except Exception:
                        x_n, y_n, v_n = 0.0, 0.0, 0
                    if x_n == 0 and y_n == 0 and v_n == 0:
                        kpts.append([0.0, 0.0, 0.0])
                    else:
                        x = max(0.0, min((x_n / 1000.0) * width, width - 1))
                        y = max(0.0, min((y_n / 1000.0) * height, height - 1))
                        if v_n not in (0, 1, 2):
                            v_n = 0
                        kpts.append([x, y, float(v_n)])
            elif len(pose_nums) >= 34:
                for i in range(17):
                    try:
                        x_n = float(pose_nums[i * 2])
                        y_n = float(pose_nums[i * 2 + 1])
                    except Exception:
                        x_n, y_n = 0.0, 0.0
                    if x_n == 0 and y_n == 0:
                        kpts.append([0.0, 0.0, 0.0])
                    else:
                        x = max(0.0, min((x_n / 1000.0) * width, width - 1))
                        y = max(0.0, min((y_n / 1000.0) * height, height - 1))
                        kpts.append([x, y, 2.0])

        if len(kpts) == 17:
            poses.append({"keypoints": kpts, "score": score})
        else:
            poses.append({"keypoints": _zero_pose_17(), "score": score})

    return _align_box_pose({"box": boxes, "pose": poses}) if boxes else None


def main() -> None:
    args = parse_args()
    if not str(args.jsonl).strip():
        raise ValueError("Missing --jsonl (or yaml `jsonl`).")
    if str(args.cuda_visible_devices).strip():
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices).strip()

    add_sapiens_python_paths(REPO_ROOT)

    dataset = Stage0BoxPoseDataset(
        jsonl_path=args.jsonl,
        image_root=args.image_root if args.image_root else None,
        max_samples=args.max_samples,
    )

    eval_dataset = dataset
    if int(args.debug_sample_size) > 0 and int(args.debug_sample_size) < len(dataset):
        rng = random.Random(int(args.debug_seed))
        indices = sorted(rng.sample(range(len(dataset)), k=int(args.debug_sample_size)))
        eval_dataset = torch.utils.data.Subset(dataset, indices)

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

    model = build_model(args.model_path, get_torch_dtype(args.fp16, args.bf16))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    os.makedirs(args.output_dir, exist_ok=True)

    pred_lines: List[Dict[str, Any]] = []
    debug_lines: List[Dict[str, Any]] = []

    total_images = len(eval_dataset)
    done_images = 0
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = processor.tokenizer.eos_token_id
    if hasattr(model, 'generation_config') and model.generation_config is not None:
        model.generation_config.pad_token_id = pad_token_id
        model.generation_config.eos_token_id = processor.tokenizer.eos_token_id
    if hasattr(processor, 'generation_config') and processor.generation_config is not None:
        processor.generation_config.pad_token_id = pad_token_id
        processor.generation_config.eos_token_id = processor.tokenizer.eos_token_id

    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            total=len(dataloader),
            desc="Eval",
            dynamic_ncols=True,
        )
        for batch in pbar:
            user_texts = []
            assistant_prefixes = []
            for sample in batch:
                user_texts.append(f"<img>{sample.image_path}</img>\n{PROMPT_BOX_POSE_TAG}")
                assistant_prefixes.append("<answer>\n<person>\n<ref>person</ref>\n<box>[")

            images = [s.image for s in batch]

            # [核心修改 4]：调用分离版函数组装 Prompt
            inputs = _build_model_inputs(
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
            input_len = inputs['input_ids'].shape[1]
            gen_suffix = gen_ids[:, input_len:]
            pred_suffixes = processor.batch_decode(gen_suffix, skip_special_tokens=True)

            # [核心修改 5]：解析前先拼装 prefix + suffix
            for sample, pred_suffix, prefix in zip(batch, pred_suffixes, assistant_prefixes):
                full_pred_text = prefix + pred_suffix
                full_pred_text = _truncate_to_answer(full_pred_text)

                parsed_from = 'none'
                pred_struct_raw = _parse_tag_struct(full_pred_text, image_size=sample.image.size)
                if pred_struct_raw is not None:
                    parsed_from = 'tag'
                else:
                    pred_struct_raw = parse_json_object_from_text(full_pred_text)
                    if pred_struct_raw is not None:
                        parsed_from = 'json'

                valid_struct = pred_struct_raw is not None
                repaired_prediction_text = None
                repaired_valid = False

                if not valid_struct:
                    repaired_prediction_text = _repair_tag_once(
                        model=model,
                        processor=processor,
                        device=device,
                        raw_text=full_pred_text,
                        max_length=int(args.max_length),
                        repair_max_new_tokens=int(args.repair_max_new_tokens),
                        pad_token_id=pad_token_id,
                    )
                    repaired_prediction_text = _truncate_to_answer(repaired_prediction_text)

                    repaired_obj = _parse_tag_struct(repaired_prediction_text or "", image_size=sample.image.size)
                    if repaired_obj is None:
                        repaired_obj = parse_json_object_from_text(repaired_prediction_text or "")
                    if repaired_obj is not None:
                        pred_struct_raw = repaired_obj
                        parsed_from = 'repair'
                        repaired_valid = True
                        valid_struct = True

                if pred_struct_raw is None or not isinstance(pred_struct_raw, dict):
                    pred_struct_raw = {'box': [], 'pose': []}
                else:
                    pred_struct_raw = _align_box_pose(pred_struct_raw)

                valid_json = valid_struct

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
                pred_poses = _normalize_pose_list(pred_struct_raw.get('pose', []), image_size=(w, h))

                pred_struct_post = {'box': pred_boxes, 'pose': pred_poses}
                gt_struct = {'box': sample.gt_box, 'pose': sample.gt_pose}

                pred_lines.append(
                    {
                        'id': sample.sid,
                        'image_path': sample.image_path,
                        'prompt': PROMPT_BOX_POSE,
                        'prediction_text': full_pred_text,
                        'repaired_prediction_text': repaired_prediction_text,
                        'parsed_from': parsed_from,
                        'prediction_struct_raw': pred_struct_raw,
                        'prediction_struct_post': pred_struct_post,
                        'gt_struct': gt_struct,
                        'valid_struct': valid_struct,
                        'repaired_valid': repaired_valid,
                        'valid_json': valid_json,
                    }
                )

                pred_coco_instances = _build_prediction_coco_instances(
                    sample_id=sample.sid,
                    pred_struct_post=pred_struct_post,
                )
                debug_lines.append(
                    {
                        'id': sample.sid,
                        'image_path': sample.image_path,
                        'image_size': [w, h],
                        'parsed_from': parsed_from,
                        'valid_json': valid_json,
                        'valid_struct': valid_struct,
                        'gt_num_box': len(sample.gt_box) if isinstance(sample.gt_box, list) else 0,
                        'pred_num_box': len(pred_boxes),
                        'gt_num_pose': len(sample.gt_pose) if isinstance(sample.gt_pose, list) else 0,
                        'pred_num_pose': len(pred_poses),
                        'gt_box': sample.gt_box,
                        'gt_pose': sample.gt_pose,
                        'pred_box_post': pred_boxes,
                        'pred_pose_post': pred_poses,
                        'prediction_coco_instances': pred_coco_instances,
                        'prediction_text': full_pred_text,
                        'repaired_prediction_text': repaired_prediction_text,
                    }
                )
                pred_lines[-1]['prediction_coco_instances'] = pred_coco_instances

                # Per-image print is disabled to keep terminal output clean.
                print(json.dumps(
                    {
                        'id': sample.sid,
                        'image_path': sample.image_path,
                        'gt_box': sample.gt_box,
                        'gt_pose': sample.gt_pose,
                        'pred_box_post': pred_boxes,
                        'pred_pose_post': pred_poses,
                    },
                    ensure_ascii=False,
                    indent=2,
                ))
            done_images += len(batch)
            pbar.set_postfix_str(f"{done_images}/{total_images}")

    metrics = {
        'num_samples': len(pred_lines),
        'valid_json_rate': sum(1 for x in pred_lines if x.get('valid_json', False)) / max(1, len(pred_lines)),
        'official_primary': True,
        'pose_metric_policy': 'official_coco_only',
    }

    if not str(args.coco_ann_json).strip():
        raise ValueError('COCO official eval requires --coco-ann-json.')
    coco_metrics = evaluate_coco_official_from_predictions(
        pred_lines=pred_lines,
        ann_json=str(args.coco_ann_json).strip(),
        split_txt=str(args.coco_split_txt).strip() if str(args.coco_split_txt).strip() else None,
    )
    metrics.update(coco_metrics)

    dump_jsonl(osp.join(args.output_dir, 'predictions.jsonl'), pred_lines)
    debug_dump_file = str(args.debug_dump_file).strip()
    if not debug_dump_file:
        debug_dump_file = osp.join(args.output_dir, 'debug_samples.jsonl')
    dump_jsonl(debug_dump_file, debug_lines)

    with open(osp.join(args.output_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
