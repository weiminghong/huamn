from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.core import bbox2result

REPO_ROOT = Path(__file__).resolve().parents[2]
HQNET_ROOT = REPO_ROOT / "COCO-UniHuman-main" / "HumanQueryNet"
if str(HQNET_ROOT) not in sys.path:
    sys.path.insert(0, str(HQNET_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_dinov3_hqnet_pipeline.common.qwen_utils import build_pyramid_from_feature_map, token_sequence_to_feature_map

from models.detectors.human_query_net import HumanQueryNet
from models.builder import DETECTORS


def _stack_teacher_features(teacher_feature: Any) -> Optional[torch.Tensor]:
    if teacher_feature is None:
        return None
    if torch.is_tensor(teacher_feature):
        return teacher_feature
    if isinstance(teacher_feature, (list, tuple)) and teacher_feature:
        items = []
        for item in teacher_feature:
            if hasattr(item, "data"):
                item = item.data
            if torch.is_tensor(item):
                items.append(item)
        if items:
            return torch.stack(items, dim=0)
    return None


@DETECTORS.register_module()
class DistillHumanQueryNet(HumanQueryNet):
    """HumanQueryNet with offline Stage1 teacher-feature distillation."""

    def __init__(self, distill_cfg: Optional[Dict[str, Any]] = None, *args, **kwargs):
        self.distill_cfg = dict(distill_cfg or {})
        self.distill_enable = bool(self.distill_cfg.get("enable", False))
        self.distill_common_dim = int(self.distill_cfg.get("common_dim", 256))
        self.distill_weight = float(self.distill_cfg.get("loss_weight", 1.0))
        self.distill_levels = int(self.distill_cfg.get("num_levels", 1))
        self.distill_use_cosine = bool(self.distill_cfg.get("use_cosine", True))
        super().__init__(*args, **kwargs)
        self.teacher_proj = nn.ModuleList(
            [nn.LazyConv2d(self.distill_common_dim, kernel_size=1) for _ in range(max(1, self.distill_levels))]
        )
        self.student_proj = nn.ModuleList(
            [nn.LazyConv2d(self.distill_common_dim, kernel_size=1) for _ in range(max(1, self.distill_levels))]
        )

    def _distill_loss(self, student_feats: Iterable[torch.Tensor], teacher_feature: torch.Tensor) -> torch.Tensor:
        teacher_map = teacher_feature
        if teacher_map.ndim == 3:
            teacher_map = teacher_map.unsqueeze(0)
        elif teacher_map.ndim == 2:
            teacher_map = token_sequence_to_feature_map(teacher_map.unsqueeze(0))
        teacher_feats = build_pyramid_from_feature_map(teacher_map, num_outs=max(1, self.distill_levels))
        student_feats = list(student_feats)
        max_levels = min(len(student_feats), len(teacher_feats), len(self.teacher_proj), len(self.student_proj))
        if max_levels <= 0:
            return teacher_map.new_tensor(0.0)

        loss = teacher_map.new_tensor(0.0)
        for idx in range(max_levels):
            student = self.student_proj[idx](student_feats[-max_levels + idx])
            teacher = self.teacher_proj[idx](teacher_feats[idx].to(student.device))
            teacher = F.interpolate(teacher, size=student.shape[-2:], mode="bilinear", align_corners=False)
            loss = loss + F.l1_loss(student, teacher)
            if self.distill_use_cosine:
                student_flat = F.normalize(student.flatten(2), dim=1)
                teacher_flat = F.normalize(teacher.flatten(2), dim=1)
                loss = loss + (1.0 - (student_flat * teacher_flat).sum(dim=1).mean())
        return loss / float(max_levels)

    def forward_train(
        self,
        img,
        img_metas,
        gt_bboxes,
        gt_labels,
        gt_attributes,
        gt_keypoints,
        gt_areas,
        gt_masks,
        gt_valids,
        teacher_feature=None,
        gt_smpl_pose=None,
        gt_smpl_betas=None,
        gt_bboxes_ignore=None,
        **kwargs,
    ):
        batch_input_shape = tuple(img[0].size()[-2:])
        for img_meta in img_metas:
            img_meta["batch_input_shape"] = batch_input_shape

        student_feats = self.extract_feat(img)
        losses = self.bbox_head.forward_train(
            student_feats,
            img_metas,
            gt_bboxes,
            gt_labels,
            gt_attributes,
            gt_keypoints,
            gt_areas,
            gt_masks,
            gt_valids,
            gt_smpl_pose,
            gt_smpl_betas,
            gt_bboxes_ignore=gt_bboxes_ignore,
        )

        if self.distill_enable:
            teacher_batch = _stack_teacher_features(teacher_feature)
            if teacher_batch is not None and teacher_batch.shape[0] > 0:
                distill_loss = student_feats[0].new_tensor(0.0)
                for i in range(min(len(img_metas), teacher_batch.shape[0])):
                    sample_student = [feat[i : i + 1] for feat in student_feats]
                    distill_loss = distill_loss + self._distill_loss(sample_student, teacher_batch[i])
                losses["loss_stage1_teacher_distill"] = self.distill_weight * distill_loss / max(
                    1, min(len(img_metas), teacher_batch.shape[0])
                )
        return losses

    def simple_test(self, img, img_metas, rescale: bool = False):
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(feat, img_metas, rescale=rescale)
        result = dict()
        if "det" in results_list:
            bbox_results = [
                bbox2result(results_list["det"], results_list["label"], self.bbox_head.num_classes)
            ]
            result["det"] = bbox_results
        if "seg" in results_list:
            result["ins_results"] = self.seg2result(results_list["seg"])
        return [result]
