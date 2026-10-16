#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.data_v1 import CocoUniHumanV1Dataset
from human_lisa_qwen_v2.collator_v2 import Stage1V2Collator
from human_lisa_qwen_v3.inference_v3 import load_stage1_v3_checkpoint
from human_lisa_qwen_v2.schema_v2 import PROMPT_V2
from human_lisa_qwen_v3.train_stage1_v3 import MODEL_INPUT_KEYS
from human_lisa_qwen_v2.training_v2 import move_to_device, scalar_tensors

class ShuffledImageDataset(Dataset):
    """Keep GT records fixed while replacing only their visual evidence."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        size = len(dataset)
        self.permutation = list(range(1, size)) + ([0] if size else [])

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        target = dict(self.dataset[index])
        target["image"] = self.dataset[self.permutation[index]]["image"]
        return target


def linear_sum_assignment_square(cost: Sequence[Sequence[float]]) -> Tuple[List[int], List[int]]:
    """Dependency-free O(n^3) Hungarian assignment for square person sets."""
    size = len(cost)
    if size == 0:
        return [], []
    if any(len(row) != size for row in cost):
        raise ValueError("teacher-forced person assignment expects a square cost matrix")
    u = [0.0] * (size + 1)
    v = [0.0] * (size + 1)
    p = [0] * (size + 1)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        p[0] = row
        column0 = 0
        min_value = [float("inf")] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column0] = True
            row0 = p[column0]
            delta = float("inf")
            column1 = 0
            for column in range(1, size + 1):
                if used[column]:
                    continue
                current = float(cost[row0 - 1][column - 1]) - u[row0] - v[column]
                if current < min_value[column]:
                    min_value[column] = current
                    way[column] = column0
                if min_value[column] < delta:
                    delta = min_value[column]
                    column1 = column
            for column in range(size + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    min_value[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = way[column0]
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break
    assignment = [0] * size
    for column in range(1, size + 1):
        assignment[p[column] - 1] = column - 1
    return list(range(size)), assignment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--annotation", default="/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json")
    parser.add_argument("--image-root", default="/root/autodl-tmp/coco/train2017")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-samples", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--shuffle-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    model, processor, manifest = load_stage1_v3_checkpoint(args.checkpoint, dtype=torch.bfloat16, device=device)
    processor.tokenizer.padding_side = "right"
    dataset = CocoUniHumanV1Dataset(
        args.annotation,
        args.image_root,
        training=False,
        horizontal_flip_prob=0.0,
        max_samples=args.max_samples,
    )
    if args.shuffle_images:
        dataset = ShuffledImageDataset(dataset)
    collator = Stage1V2Collator(processor=processor, token_ids=model.token_ids, prompt=PROMPT_V2, max_length=5000)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collator, pin_memory=True)
    loss_sums: Dict[str, float] = {}
    batches = 0
    pose_visible = 0
    pose_pck_hits = 0
    pose_distance_sum = 0.0
    pose_visibility_correct = 0
    pose_visibility_total = 0
    pose_hungarian_visible = 0
    pose_hungarian_pck_hits = 0
    pose_hungarian_distance_sum = 0.0
    pose_person_spread_sum = 0.0
    pose_target_person_spread_sum = 0.0
    pose_spread_images = 0
    mask_people = 0
    mask_iou_sum = 0.0
    mask_dice_sum = 0.0
    mask_hungarian_iou_sum = 0.0
    mask_hungarian_dice_sum = 0.0
    query_persons = 0
    query_box_iou_sum = 0.0
    query_box_iou50_hits = 0
    query_positive_objectness_sum = 0.0

    with torch.no_grad():
        for raw in loader:
            batch = move_to_device({name: value for name, value in raw.items() if name in MODEL_INPUT_KEYS}, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch, return_predictions=True)
            for name, value in scalar_tensors(outputs).items():
                loss_sums[name] = loss_sums.get(name, 0.0) + value
            batches += 1
            for index, prediction in enumerate(outputs["predictions"]):
                query_boxes = prediction["matched_query_boxes_xyxy"].float()
                target_boxes_all = batch["box_targets"][index].float()
                if query_boxes.shape != target_boxes_all.shape:
                    raise RuntimeError("matched query box count differs from GT person count")
                top_left = torch.maximum(query_boxes[:, :2], target_boxes_all[:, :2])
                bottom_right = torch.minimum(query_boxes[:, 2:], target_boxes_all[:, 2:])
                intersection = (bottom_right - top_left).clamp(min=0).prod(dim=-1)
                query_area = (query_boxes[:, 2:] - query_boxes[:, :2]).clamp(min=0).prod(dim=-1)
                target_area = (target_boxes_all[:, 2:] - target_boxes_all[:, :2]).clamp(min=0).prod(dim=-1)
                query_iou = intersection / (query_area + target_area - intersection).clamp(min=1e-7)
                query_persons += int(query_iou.numel())
                query_box_iou_sum += float(query_iou.sum())
                query_box_iou50_hits += int((query_iou >= 0.5).sum())
                query_positive_objectness_sum += float(prediction["matched_query_objectness"].sigmoid().sum())

                target_pose = batch["pose_targets"][index].float()
                predicted_pose = prediction["pose_xy"].float()
                visibility = target_pose[..., 2].long().clamp(0, 2)
                visible = visibility > 0
                pose_indices = batch["pose_person_indices"][index]
                boxes = batch["box_targets"][index].float().index_select(0, pose_indices)
                scale = torch.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]).clamp(min=1e-4)
                distance = (predicted_pose - target_pose[..., :2]).square().sum(dim=-1).sqrt() / scale[:, None]
                pose_visible += int(visible.sum())
                pose_pck_hits += int((distance[visible] <= 0.2).sum())
                pose_distance_sum += float(distance[visible].sum())
                predicted_visibility = prediction["pose_visibility"].argmax(dim=-1)
                pose_visibility_correct += int((predicted_visibility == visibility).sum())
                pose_visibility_total += int(visibility.numel())

                people = int(predicted_pose.shape[0])
                if people > 1:
                    pose_person_spread_sum += float(predicted_pose.flatten(1).std(dim=0, unbiased=False).mean())
                    pose_target_person_spread_sum += float(target_pose[..., :2].flatten(1).std(dim=0, unbiased=False).mean())
                    pose_spread_images += 1
                pair_cost = torch.empty((people, people), device=predicted_pose.device, dtype=torch.float32)
                pair_distances = []
                for pred_index in range(people):
                    pair_distance = (
                        predicted_pose[pred_index][None] - target_pose[..., :2]
                    ).square().sum(dim=-1).sqrt() / scale[:, None]
                    pair_distances.append(pair_distance)
                    for target_index in range(people):
                        target_visible = visible[target_index]
                        pair_cost[pred_index, target_index] = (
                            pair_distance[target_index][target_visible].mean() if target_visible.any() else 0.0
                        )
                pred_order, target_order = linear_sum_assignment_square(pair_cost.detach().cpu().tolist())
                for pred_index, target_index in zip(pred_order, target_order):
                    target_visible = visible[target_index]
                    pair_distance = pair_distances[pred_index][target_index]
                    pose_hungarian_visible += int(target_visible.sum())
                    pose_hungarian_pck_hits += int((pair_distance[target_visible] <= 0.2).sum())
                    pose_hungarian_distance_sum += float(pair_distance[target_visible].sum())

                logits = prediction["mask_logits"].float()
                targets = batch["seg_targets"][index].float().unsqueeze(1)
                targets = F.interpolate(targets, size=logits.shape[-2:], mode="nearest")[:, 0] > 0.5
                predicted = logits.sigmoid() >= 0.5
                intersection = (predicted & targets).flatten(1).sum(dim=1).float()
                union = (predicted | targets).flatten(1).sum(dim=1).float().clamp(min=1)
                denominator = predicted.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
                mask_iou_sum += float((intersection / union).sum())
                mask_dice_sum += float(((2.0 * intersection) / denominator.clamp(min=1)).sum())
                mask_people += int(logits.shape[0])
                predicted_flat = predicted.flatten(1).float()
                target_flat = targets.flatten(1).float()
                pair_intersection = predicted_flat @ target_flat.transpose(0, 1)
                pair_denominator = predicted_flat.sum(dim=1)[:, None] + target_flat.sum(dim=1)[None, :]
                pair_union = pair_denominator - pair_intersection
                pair_iou = pair_intersection / pair_union.clamp(min=1)
                pair_dice = (2.0 * pair_intersection) / pair_denominator.clamp(min=1)
                pred_order, target_order = linear_sum_assignment_square((1.0 - pair_iou).detach().cpu().tolist())
                mask_hungarian_iou_sum += sum(
                    float(pair_iou[pred_index, target_index])
                    for pred_index, target_index in zip(pred_order, target_order)
                )
                mask_hungarian_dice_sum += sum(
                    float(pair_dice[pred_index, target_index])
                    for pred_index, target_index in zip(pred_order, target_order)
                )

    metrics: Dict[str, Any] = {
        "protocol": "stage1_v3_teacher_forced_gt_records_shuffled_image_diagnostic" if args.shuffle_images else "stage1_v3_teacher_forced_gt_records_query_matched_diagnostic",
        "images": len(dataset),
        "checkpoint": args.checkpoint,
        "query_persons": query_persons,
        "query_box_mean_iou": query_box_iou_sum / max(1, query_persons),
        "query_box_iou50_rate": query_box_iou50_hits / max(1, query_persons),
        "query_positive_objectness_mean": query_positive_objectness_sum / max(1, query_persons),
        "pose_visible_keypoints": pose_visible,
        "pose_pck@0.2_gt_box": pose_pck_hits / max(1, pose_visible),
        "pose_mean_normalized_error_gt_box": pose_distance_sum / max(1, pose_visible),
        "pose_pck@0.2_gt_box_hungarian_person": pose_hungarian_pck_hits / max(1, pose_hungarian_visible),
        "pose_mean_normalized_error_gt_box_hungarian_person": pose_hungarian_distance_sum / max(1, pose_hungarian_visible),
        "pose_visibility_accuracy": pose_visibility_correct / max(1, pose_visibility_total),
        "pose_predicted_person_spread": pose_person_spread_sum / max(1, pose_spread_images),
        "pose_target_person_spread": pose_target_person_spread_sum / max(1, pose_spread_images),
        "seg_people": mask_people,
        "seg_binary_miou": mask_iou_sum / max(1, mask_people),
        "seg_binary_dice": mask_dice_sum / max(1, mask_people),
        "seg_binary_miou_hungarian_person": mask_hungarian_iou_sum / max(1, mask_people),
        "seg_binary_dice_hungarian_person": mask_hungarian_dice_sum / max(1, mask_people),
        "manifest": manifest,
    }
    metrics.update({name: value / max(1, batches) for name, value in loss_sums.items()})
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
