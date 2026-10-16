from __future__ import annotations

import torch
from mmdet.models.task_modules.samplers import SamplingResult


class LegacyPseudoSampler:
    """Tensor-style PseudoSampler compatible with MMDet2 head code."""

    def sample(self, assign_result, bboxes: torch.Tensor, gt_bboxes: torch.Tensor, *args, **kwargs):
        pos_inds = torch.nonzero(assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        return SamplingResult(
            pos_inds=pos_inds,
            neg_inds=neg_inds,
            priors=bboxes,
            gt_bboxes=gt_bboxes,
            assign_result=assign_result,
            gt_flags=gt_flags,
            avg_factor_with_neg=False,
        )
