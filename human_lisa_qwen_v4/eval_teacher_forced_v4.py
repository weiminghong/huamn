#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v1.official_eval_v1 import (
    encode_binary_mask,
    evaluate_all_official,
)

from .collator_v4 import Stage1V4Collator
from .inference_v4 import load_stage1_v4_checkpoint
from .schema_v4 import PROMPT_V4
from .training_v4 import move_to_device


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ATTR_EVAL = (
    REPO_ROOT
    / "HumanQueryNet_mmdet3_port"
    / "datasets"
    / "api_wrappers"
    / "eval_attr.py"
)
METADATA_KEYS = {
    "label_audit",
    "sample_ids",
    "target_texts",
    "source_ann_ids",
    "sequence_lengths",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--annotation",
        default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json",
    )
    parser.add_argument(
        "--image-root", default="/root/autodl-tmp/coco/val2017"
    )
    parser.add_argument("--base-model", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=5000)
    parser.add_argument("--attr-eval-source", default=str(DEFAULT_ATTR_EVAL))
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def _scalar_outputs(outputs: Dict[str, Any]) -> Dict[str, float]:
    return {
        name: float(value.detach().float().cpu().item())
        for name, value in outputs.items()
        if torch.is_tensor(value) and value.numel() == 1
    }


def _pose_record(
    image_id: int,
    person_index: int,
    pose_xy: torch.Tensor,
    pose_visibility: torch.Tensor,
    width: int,
    height: int,
) -> Dict[str, Any]:
    keypoints: List[float] = []
    for joint in range(17):
        keypoints.extend(
            [
                float(pose_xy[joint, 0] * width),
                float(pose_xy[joint, 1] * height),
                float(pose_visibility[joint]),
            ]
        )
    return {
        "image_id": int(image_id),
        "category_id": 1,
        "person_id": int(person_index + 1),
        "score": 1.0,
        "keypoints": keypoints,
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.max_length != 5000:
        raise ValueError(
            "v4 teacher-forced eval requires batch_size>0 and length=5000"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("v4 teacher-forced evaluation requires CUDA")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model, processor, manifest = load_stage1_v4_checkpoint(
        args.checkpoint,
        base_model_path=args.base_model or None,
        dtype=dtype,
        device=device,
    )
    dataset = CocoUniHumanV1Dataset(
        args.annotation,
        args.image_root,
        training=False,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples,
        decode_eval_segmentation=True,
    )
    collator = Stage1V4Collator(
        processor=processor,
        token_ids=model.token_ids,
        prompt=PROMPT_V4,
        max_length=args.max_length,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=collator,
        drop_last=False,
    )
    records: List[Dict[str, Any]] = []
    loss_sums: Dict[str, float] = {}
    loss_batches = 0
    pose_hits = 0
    pose_total = 0
    mask_ious: List[float] = []
    mask_dices: List[float] = []
    pose_labels = 0
    seg_labels = 0
    status_path = output / "status.json"

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            metadata = {name: batch[name] for name in METADATA_KEYS}
            model_batch = {
                name: value
                for name, value in batch.items()
                if name not in METADATA_KEYS
            }
            model_batch = move_to_device(model_batch, device)
            with torch.autocast(
                "cuda", dtype=dtype, enabled=(dtype == torch.bfloat16)
            ):
                outputs = model(**model_batch, return_predictions=True)
            for name, value in _scalar_outputs(outputs).items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value
            loss_batches += 1

            for row, prediction in enumerate(outputs["predictions"]):
                image_id = int(metadata["sample_ids"][row])
                height, width = [
                    int(value) for value in model_batch["original_sizes"][row]
                ]
                target_obj = json.loads(metadata["target_texts"][row])
                records_by_person: Dict[int, Dict[str, Any]] = {}

                pose_indices = prediction["pose_person_indices"].detach().cpu()
                pose_xy = prediction["pose_coordinates"].detach().float().cpu()
                pose_visibility = prediction[
                    "pose_visibility_logits"
                ].float().softmax(dim=-1)[..., 1:].sum(dim=-1).cpu()
                pose_target = model_batch["pose_targets"][row].detach().float().cpu()
                pose_labels += int(len(pose_indices))
                for local_index, index_tensor in enumerate(pose_indices):
                    person_index = int(index_tensor)
                    records_by_person[person_index] = _pose_record(
                        image_id,
                        person_index,
                        pose_xy[local_index],
                        pose_visibility[local_index],
                        width,
                        height,
                    )
                    visible = pose_target[local_index, :, 2].gt(0)
                    if visible.any():
                        predicted_pixels = pose_xy[local_index].clone()
                        predicted_pixels[:, 0] *= width
                        predicted_pixels[:, 1] *= height
                        target_pixels = pose_target[local_index, :, :2].clone()
                        target_pixels[:, 0] *= width
                        target_pixels[:, 1] *= height
                        distances = torch.linalg.vector_norm(
                            predicted_pixels - target_pixels, dim=-1
                        )
                        box = target_obj["persons"][person_index]["box"]
                        box_width = (
                            float(box[2]) - float(box[0])
                        ) * width / 1000.0
                        box_height = (
                            float(box[3]) - float(box[1])
                        ) * height / 1000.0
                        threshold = 0.2 * max(box_width, box_height, 1.0)
                        pose_hits += int((distances[visible] <= threshold).sum())
                        pose_total += int(visible.sum())

                seg_indices = prediction["seg_person_indices"].detach().cpu()
                mask_logits = prediction["mask_logits"].detach().float()
                mask_targets = model_batch["seg_targets"][row].detach().float()
                seg_labels += int(len(seg_indices))
                for local_index, index_tensor in enumerate(seg_indices):
                    person_index = int(index_tensor)
                    resized = F.interpolate(
                        mask_logits[local_index : local_index + 1, None],
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]
                    predicted_mask = resized.sigmoid().ge(0.5)
                    target_mask = mask_targets[local_index].bool()
                    intersection = int((predicted_mask & target_mask).sum())
                    union = int((predicted_mask | target_mask).sum())
                    predicted_area = int(predicted_mask.sum())
                    target_area = int(target_mask.sum())
                    mask_ious.append(intersection / max(1, union))
                    mask_dices.append(
                        2.0 * intersection / max(1, predicted_area + target_area)
                    )
                    record = records_by_person.setdefault(
                        person_index,
                        {
                            "image_id": image_id,
                            "category_id": 1,
                            "person_id": person_index + 1,
                            "score": 1.0,
                        },
                    )
                    record["segmentation"] = encode_binary_mask(
                        predicted_mask.cpu().numpy()
                    )
                records.extend(records_by_person.values())

            status = {
                "processed": min(
                    len(dataset), (batch_index + 1) * args.batch_size
                ),
                "total": len(dataset),
                "pose_labels": pose_labels,
                "seg_labels": seg_labels,
                "pose_pck_bbox_0.2": pose_hits / max(1, pose_total),
                "seg_miou": sum(mask_ious) / max(1, len(mask_ious)),
            }
            status_path.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            print(json.dumps(status), flush=True)

    image_ids = [int(image["id"]) for image in dataset.images]
    official = evaluate_all_official(
        args.annotation,
        records,
        image_ids=image_ids,
        attr_eval_source=Path(args.attr_eval_source),
    )
    metrics = {
        "images": len(dataset),
        "pose_labels": pose_labels,
        "seg_labels": seg_labels,
        "pose_pck_bbox_0.2": pose_hits / max(1, pose_total),
        "pose_visible_keypoints": pose_total,
        "seg_miou": sum(mask_ious) / max(1, len(mask_ious)),
        "seg_dice": sum(mask_dices) / max(1, len(mask_dices)),
        "pose_AP": official.get("keypoints_AP", 0.0),
        "pose_AP50": official.get("keypoints_AP50", 0.0),
        "pose_AP75": official.get("keypoints_AP75", 0.0),
        "segm_AP": official.get("segm_AP", 0.0),
        "segm_AP50": official.get("segm_AP50", 0.0),
        "segm_AP75": official.get("segm_AP75", 0.0),
        "mean_losses": {
            name: value / max(1, loss_batches)
            for name, value in sorted(loss_sums.items())
        },
        "checkpoint": args.checkpoint,
        "annotation": args.annotation,
        "split": (
            "full_val" if args.max_samples is None else "diagnostic_val_subset"
        ),
        "protocol": (
            "stage1_v4_teacher_forced_gt_person_records_oracle_binding_"
            "official_pose_segm_diagnostic"
        ),
        "manifest": manifest,
    }
    (output / "predictions.teacher_forced_pose_seg.json").write_text(
        json.dumps(records), encoding="utf-8"
    )
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
