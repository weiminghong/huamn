from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoProcessor

# lm-format-enforcer<=0.11 imports PreTrainedTokenizerBase from
# transformers.tokenization_utils, while transformers 5 exposes it from
# tokenization_utils_base. Keep guided JSON decoding semantics unchanged.
try:
    import transformers.tokenization_utils as _tokenization_utils
    from transformers.tokenization_utils_base import (
        PreTrainedTokenizerBase as _PreTrainedTokenizerBase,
    )
    if not hasattr(_tokenization_utils, "PreTrainedTokenizerBase"):
        _tokenization_utils.PreTrainedTokenizerBase = _PreTrainedTokenizerBase
except Exception:
    pass

PKG_DIR = "/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs"
if PKG_DIR not in sys.path:
    sys.path.insert(0, PKG_DIR)

from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)

from .collator_v1 import _find_last_subsequence
from .heads_v1 import aligned_box_iou, roi_pool_visual_tokens
from .modeling import load_qwen_vl_backbone
from .modeling_stage1_v1 import Stage1V1Model, _parent_states
from .official_eval_v1 import encode_binary_mask
from .schema_v1 import (
    GUIDED_JSON_SCHEMA_V1,
    PROMPT_V1,
    canonicalize_person_ids_v1,
    validate_prediction_object_v1,
)
from .train_stage1_v1 import add_special_tokens, hidden_size_from_config


def combine_detection_score(token_probability: float, quality: float) -> float:
    token_probability = min(1.0, max(0.0, float(token_probability)))
    quality = min(1.0, max(0.0, float(quality)))
    return math.sqrt(token_probability * quality)


def _box_iou(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        aligned_box_iou(first.reshape(1, 4), second.reshape(1, 4))[0].item()
    )


def _record_iou(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    first_x, first_y, first_w, first_h = [float(value) for value in first["bbox"]]
    second_x, second_y, second_w, second_h = [float(value) for value in second["bbox"]]
    first_x2, first_y2 = first_x + max(0.0, first_w), first_y + max(0.0, first_h)
    second_x2, second_y2 = second_x + max(0.0, second_w), second_y + max(0.0, second_h)
    intersection = max(0.0, min(first_x2, second_x2) - max(first_x, second_x))
    intersection *= max(0.0, min(first_y2, second_y2) - max(first_y, second_y))
    union = max(0.0, first_w) * max(0.0, first_h)
    union += max(0.0, second_w) * max(0.0, second_h) - intersection
    return intersection / union if union > 0.0 else 0.0


def suppress_near_duplicate_person_records(
    records: Sequence[Dict[str, Any]], *, iou_threshold: float = 0.90
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Apply score-aware NMS only to near-identical generated person records.

    This is not a person cap: any number of spatially distinct people remains.
    Suppression operates on the already assembled record, preserving box, pose,
    segmentation and attributes as one atomic person prediction.
    """
    if not 0.0 < float(iou_threshold) <= 1.0:
        raise ValueError("near-duplicate IoU threshold must be in (0, 1]")
    ordered = sorted(
        enumerate(records),
        key=lambda item: (-float(item[1].get("score", 0.0)), item[0]),
    )
    kept_indices: List[int] = []
    suppressed: List[Dict[str, Any]] = []
    for index, record in ordered:
        matching_kept = next(
            (
                kept_index
                for kept_index in kept_indices
                if _record_iou(record, records[kept_index]) >= float(iou_threshold)
            ),
            None,
        )
        if matching_kept is None:
            kept_indices.append(index)
        else:
            suppressed.append(
                {
                    "person_id": int(record["person_id"]),
                    "kept_person_id": int(records[matching_kept]["person_id"]),
                    "iou": _record_iou(record, records[matching_kept]),
                }
            )
    kept_indices.sort()
    return (
        [records[index] for index in kept_indices],
        {
            "iou_threshold": float(iou_threshold),
            "raw_records": int(len(records)),
            "kept_records": int(len(kept_indices)),
            "suppressed_records": suppressed,
        },
    )


def semantic_health(
    obj: Dict[str, Any],
    *,
    boxes: torch.Tensor,
    pose_count: int,
    seg_count: int,
    eos_seen: bool,
    raw_token_count: int,
) -> Dict[str, Any]:
    errors: List[str] = []
    persons = obj.get("persons", []) if isinstance(obj, dict) else []
    expected_boxes = len(persons)
    expected_pose = sum(person.get("pose") == "<pose>" for person in persons)
    expected_seg = sum(person.get("seg") == "<seg>" for person in persons)
    if boxes.shape != (expected_boxes, 4):
        errors.append("box_count_mismatch")
    if int(pose_count) != int(expected_pose):
        errors.append("pose_count_mismatch")
    if int(seg_count) != int(expected_seg):
        errors.append("seg_count_mismatch")
    if not eos_seen:
        errors.append("missing_eos")
    if int(raw_token_count) >= 5000:
        errors.append("generation_hit_token_cap")
    if not torch.isfinite(boxes).all():
        errors.append("nonfinite_boxes")
    if boxes.numel() and (
        (boxes[:, 2] <= boxes[:, 0]).any()
        or (boxes[:, 3] <= boxes[:, 1]).any()
        or (boxes < 0).any()
        or (boxes > 1).any()
    ):
        errors.append("invalid_box_geometry")
    duplicate_pairs = 0
    for first in range(len(boxes)):
        for second in range(first + 1, len(boxes)):
            if _box_iou(boxes[first], boxes[second]) >= 0.9:
                duplicate_pairs += 1
    if duplicate_pairs:
        errors.append("duplicate_boxes")
    return {
        "ok": not errors,
        "errors": errors,
        "persons": expected_boxes,
        "expected_pose": expected_pose,
        "expected_seg": expected_seg,
        "duplicate_pairs": duplicate_pairs,
        "eos_seen": bool(eos_seen),
        "raw_token_count": int(raw_token_count),
    }



def summarize_inference_audit(
    results: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    total = len(results)
    denominator = max(1, total)
    return {
        "images": total,
        "schema_ok_rate": sum(
            int(result.get("schema_ok", False)) for result in results
        )
        / denominator,
        "raw_schema_ok_rate": sum(
            int(result.get("raw_schema_ok", False)) for result in results
        )
        / denominator,
        "person_id_repair_image_rate": sum(
            int(int(result.get("person_id_repairs", 0)) > 0)
            for result in results
        )
        / denominator,
        "person_id_repairs": sum(
            int(result.get("person_id_repairs", 0)) for result in results
        ),
        "health_ok_rate": sum(
            int(result.get("health", {}).get("ok", False))
            for result in results
        )
        / denominator,
        "eos_rate": sum(
            int(result.get("eos_seen", False)) for result in results
        )
        / denominator,
    }

def assemble_coco_records(
    *,
    image_id: int,
    width: int,
    height: int,
    obj: Dict[str, Any],
    boxes: torch.Tensor,
    quality: torch.Tensor,
    token_probabilities: torch.Tensor,
    pose_xy: torch.Tensor,
    pose_visibility_logits: torch.Tensor,
    mask_logits: torch.Tensor,
) -> List[Dict[str, Any]]:
    persons = obj["persons"]
    if boxes.shape[0] != len(persons):
        raise RuntimeError("box count does not match person records")
    if quality.shape[0] != len(persons):
        raise RuntimeError("quality count does not match person records")
    if token_probabilities.shape[0] != len(persons):
        raise RuntimeError("token probability count does not match records")
    records: List[Dict[str, Any]] = []
    pose_index = 0
    seg_index = 0
    for person_index, person in enumerate(persons):
        box = boxes[person_index].detach().float().cpu()
        x1 = float(box[0] * width)
        y1 = float(box[1] * height)
        x2 = float(box[2] * width)
        y2 = float(box[3] * height)
        score = combine_detection_score(
            float(token_probabilities[person_index]),
            float(quality[person_index]),
        )
        record: Dict[str, Any] = {
            "image_id": int(image_id),
            "category_id": 1,
            "person_id": int(person["person_id"]),
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": score,
            "gender": int(person["gender"]),
            "age": float(person["age"]),
        }
        if person["pose"] == "<pose>":
            if pose_index >= pose_xy.shape[0]:
                raise RuntimeError("pose output count is too small")
            xy = pose_xy[pose_index].detach().float().cpu()
            visibility = pose_visibility_logits[pose_index].argmax(dim=-1)
            keypoints: List[float] = []
            for joint in range(17):
                visible = int(visibility[joint].item())
                keypoints.extend(
                    [
                        float(xy[joint, 0] * width),
                        float(xy[joint, 1] * height),
                        float(visible),
                    ]
                )
            record["keypoints"] = keypoints
            pose_index += 1
        if person["seg"] == "<seg>":
            if seg_index >= mask_logits.shape[0]:
                raise RuntimeError("seg output count is too small")
            logits = mask_logits[seg_index : seg_index + 1, None].float()
            resized = F.interpolate(
                logits,
                size=(int(height), int(width)),
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            record["segmentation"] = encode_binary_mask(
                resized.sigmoid().ge(0.5).detach().cpu().numpy()
            )
            seg_index += 1
        records.append(record)
    if pose_index != pose_xy.shape[0] or seg_index != mask_logits.shape[0]:
        raise RuntimeError("unused pose or segmentation outputs")
    return records


def build_chat_prefix(processor: Any, prompt: str = PROMPT_V1) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def make_guided_fn(tokenizer: Any, prompt_width: int):
    parser = JsonSchemaParser(GUIDED_JSON_SCHEMA_V1)
    base_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

    def allowed(batch_id: int, sent: torch.Tensor):
        return base_fn(batch_id, sent[prompt_width:])

    return allowed


def decode_generated_json(
    tokenizer: Any, generated_ids: torch.Tensor
) -> Tuple[str, bool]:
    token_list = [int(value) for value in generated_ids.tolist()]
    eos_ids = {
        value
        for value in [
            getattr(tokenizer, "eos_token_id", None),
            tokenizer.convert_tokens_to_ids("<|im_end|>"),
        ]
        if value is not None and int(value) >= 0
    }
    eos_seen = any(value in eos_ids for value in token_list)
    text = tokenizer.decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    for token in [
        getattr(tokenizer, "eos_token", None),
        "<|im_end|>",
        "<|endoftext|>",
    ]:
        if token and token in text:
            text = text.split(token, 1)[0]
    return text.strip(), eos_seen




def _pose_decoder_type_from_checkpoint(checkpoint: Path) -> str:
    state = torch.load(
        checkpoint / "stage1_task_heads.pt",
        map_location="cpu",
        weights_only=False,
    )
    decoder_type = str(state.get("pose_decoder_type", "coordinate"))
    if decoder_type not in {"direct_joint_query", "coordinate", "topdown_heatmap"}:
        raise RuntimeError(
            f"unsupported pose decoder type in checkpoint: {decoder_type}"
        )
    return decoder_type
def load_stage1_checkpoint(
    checkpoint_dir: str,
    *,
    base_model_path: Optional[str] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> Tuple[Stage1V1Model, Any, Dict[str, Any]]:
    checkpoint = Path(checkpoint_dir)
    manifest_path = checkpoint / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.is_file()
        else {}
    )
    base_model_path = base_model_path or manifest.get(
        "model", "/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct"
    )
    processor = AutoProcessor.from_pretrained(
        checkpoint / "processor", trust_remote_code=True
    )
    base = load_qwen_vl_backbone(base_model_path, dtype)
    token_ids = add_special_tokens(processor, base)
    from peft import PeftModel

    base = PeftModel.from_pretrained(
        base, checkpoint / "base_or_adapter", is_trainable=False
    )
    wrapper = Stage1V1Model(
        base_model=base,
        hidden_size=hidden_size_from_config(base),
        token_ids=token_ids,
        image_token_id=int(
            getattr(getattr(base, "config", None), "image_token_id", 151655)
        ),
        decoder_dim=int(manifest.get("decoder_dim", 256)),
        decoder_heads=int(manifest.get("decoder_heads", 8)),
        pose_decoder_type=_pose_decoder_type_from_checkpoint(checkpoint),
        visual_merge_size=2,
    )
    wrapper.load_task_heads(str(checkpoint / "stage1_task_heads.pt"))
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wrapper.to(device)
    wrapper.eval()
    return wrapper, processor, manifest


def _target_token_probability_from_hidden(
    model: torch.nn.Module,
    hidden: torch.Tensor,
    positions: List[int],
    token_id: int,
) -> torch.Tensor:
    if any(int(position) <= 0 for position in positions):
        raise RuntimeError("special token has no preceding prediction hidden state")
    if not positions:
        return hidden.new_zeros((0,), dtype=torch.float32)
    preceding = torch.tensor(
        [int(position) - 1 for position in positions],
        device=hidden.device,
        dtype=torch.long,
    )
    rows = hidden[0].index_select(0, preceding)
    output_head = model.get_output_embeddings()
    logits = output_head(rows).float()
    return logits.softmax(dim=-1)[:, int(token_id)]


@torch.no_grad()
def decode_heads_for_text(
    model: Stage1V1Model,
    processor: Any,
    *,
    image: Any,
    raw_text: str,
    obj: Dict[str, Any],
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    prefix = build_chat_prefix(processor)
    eos = processor.tokenizer.eos_token or ""
    target_text = raw_text + eos
    processor.tokenizer.padding_side = "right"
    inputs = processor(
        text=[prefix + target_text],
        images=[image],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    target_ids = processor.tokenizer(
        target_text, add_special_tokens=False
    )["input_ids"]
    sequence = [int(value) for value in inputs["input_ids"][0].tolist()]
    target_start = _find_last_subsequence(sequence, target_ids)
    if target_start < 0:
        raise RuntimeError("cannot locate generated JSON in multimodal sequence")
    with torch.autocast(
        "cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype == torch.bfloat16)
    ):
        outputs = model.base_model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
            logits_to_keep=1,
        )
        hidden = outputs.hidden_states[-1]
        visual_tokens, spatial_shape = model._visual_tokens_and_shape(
            inputs["input_ids"],
            hidden,
            inputs["image_grid_thw"],
            0,
        )
        target_positions = torch.arange(
            inputs["input_ids"].shape[1], device=device
        ).ge(target_start)
        box_positions = (
            inputs["input_ids"][0].eq(model.token_ids["box"])
            & target_positions
        ).nonzero(as_tuple=False).flatten()
        pose_positions = (
            inputs["input_ids"][0].eq(model.token_ids["pose"])
            & target_positions
        ).nonzero(as_tuple=False).flatten()
        seg_positions = (
            inputs["input_ids"][0].eq(model.token_ids["seg"])
            & target_positions
        ).nonzero(as_tuple=False).flatten()
        box_states = hidden[0, box_positions]
        pose_states = hidden[0, pose_positions]
        seg_states = hidden[0, seg_positions]
        persons = obj["persons"]
        if len(box_states) != len(persons):
            raise RuntimeError("generated box token count does not match records")
        boxes, quality_logits = model.box_head(box_states)
        pose_person_indices = torch.tensor(
            [
                index
                for index, person in enumerate(persons)
                if person["pose"] == "<pose>"
            ],
            device=device,
            dtype=torch.long,
        )
        seg_person_indices = torch.tensor(
            [
                index
                for index, person in enumerate(persons)
                if person["seg"] == "<seg>"
            ],
            device=device,
            dtype=torch.long,
        )
        pose_parent = _parent_states(
            box_states, pose_person_indices, pose_states, "pose"
        )
        pose_boxes = boxes.index_select(0, pose_person_indices)
        if model.pose_decoder_type == "direct_joint_query":
            pose_visual_tokens = visual_tokens.unsqueeze(0)
            pose_box_condition = pose_boxes
        elif model.pose_roi_enabled:
            pose_visual_tokens = roi_pool_visual_tokens(
                visual_tokens,
                spatial_shape=spatial_shape,
                boxes=pose_boxes,
                output_size=7,
            )
            pose_box_condition = pose_boxes
        else:
            pose_visual_tokens = visual_tokens.unsqueeze(0)
            pose_box_condition = None
        if model.pose_decoder_type == "direct_joint_query":
            pose_xy, pose_visibility = model.pose_decoder(
                pose_states,
                pose_parent,
                pose_visual_tokens,
                spatial_shape=spatial_shape,
                person_boxes=pose_box_condition,
            )
        elif model.pose_decoder_type == "topdown_heatmap":
            if pose_box_condition is None:
                raise RuntimeError("topdown heatmap pose requires ROI boxes")
            pose_xy, pose_visibility, _ = model.pose_decoder(
                pose_states,
                pose_parent,
                pose_visual_tokens,
                person_boxes=pose_box_condition,
            )
        else:
            pose_xy, pose_visibility = model.pose_decoder(
                pose_states,
                pose_parent,
                pose_visual_tokens,
                person_boxes=pose_box_condition,
                use_joint_reference=model.pose_joint_reference_enabled,
            )
        seg_parent = _parent_states(
            box_states, seg_person_indices, seg_states, "seg"
        )
        mask_logits = model.mask_decoder(
            seg_states,
            seg_parent,
            visual_tokens.unsqueeze(0),
            spatial_shape=spatial_shape,
        )
    token_probabilities = _target_token_probability_from_hidden(
        model.base_model,
        hidden,
        [int(value) for value in box_positions.tolist()],
        model.token_ids["box"],
    )
    return {
        "boxes": boxes,
        "quality": torch.sigmoid(quality_logits),
        "token_probabilities": token_probabilities,
        "pose_xy": pose_xy,
        "pose_visibility_logits": pose_visibility,
        "mask_logits": mask_logits,
        "raw_token_count": len(target_ids),
    }


@torch.no_grad()
def infer_samples(
    model: Stage1V1Model,
    processor: Any,
    samples: Sequence[Dict[str, Any]],
    *,
    max_new_tokens: int = 5000,
    dtype: torch.dtype = torch.bfloat16,
    device: Optional[torch.device] = None,
) -> List[Dict[str, Any]]:
    if not samples:
        return []
    device = device or next(model.parameters()).device
    prefix = build_chat_prefix(processor)
    processor.tokenizer.padding_side = "left"
    inputs = processor(
        text=[prefix] * len(samples),
        images=[sample["image"] for sample in samples],
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(device)
    prompt_width = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.autocast(
        "cuda", dtype=dtype, enabled=(device.type == "cuda" and dtype == torch.bfloat16)
    ):
        generation = model.base_model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            prefix_allowed_tokens_fn=make_guided_fn(
                processor.tokenizer, prompt_width
            ),
            eos_token_id=processor.tokenizer.eos_token_id,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )
    elapsed = time.perf_counter() - started
    results: List[Dict[str, Any]] = []
    for index, sample in enumerate(samples):
        generated_ids = generation[index, prompt_width:]
        raw, eos_seen = decode_generated_json(
            processor.tokenizer, generated_ids
        )
        result: Dict[str, Any] = {
            "image_id": int(sample["id"]),
            "raw": raw,
            "eos_seen": eos_seen,
            "schema_ok": False,
            "raw_schema_ok": False,
            "person_id_repairs": 0,
            "records": [],
            "latency_seconds": elapsed / len(samples),
        }
        try:
            obj = json.loads(raw)
            person_id_repairs = canonicalize_person_ids_v1(obj)
            validate_prediction_object_v1(obj)
            decoded = decode_heads_for_text(
                model,
                processor,
                image=sample["image"],
                raw_text=raw,
                obj=obj,
                dtype=dtype,
                device=device,
            )
            health = semantic_health(
                obj,
                boxes=decoded["boxes"],
                pose_count=decoded["pose_xy"].shape[0],
                seg_count=decoded["mask_logits"].shape[0],
                eos_seen=eos_seen,
                raw_token_count=decoded["raw_token_count"],
            )
            raw_records = assemble_coco_records(
                image_id=int(sample["id"]),
                width=int(sample["width"]),
                height=int(sample["height"]),
                obj=obj,
                **{
                    name: decoded[name]
                    for name in [
                        "boxes",
                        "quality",
                        "token_probabilities",
                        "pose_xy",
                        "pose_visibility_logits",
                        "mask_logits",
                    ]
                },
            )
            records, duplicate_suppression = suppress_near_duplicate_person_records(
                raw_records
            )
            result.update(
                {
                    "schema_ok": True,
                    "raw_schema_ok": person_id_repairs == 0,
                    "person_id_repairs": person_id_repairs,
                    "health": health,
                    "raw_record_count": len(raw_records),
                    "duplicate_suppression": duplicate_suppression,
                    "records": records,
                }
            )
        except Exception as error:
            result["error"] = repr(error)
            result["health"] = {
                "ok": False,
                "errors": ["inference_exception"],
            }
        results.append(result)
    return results

