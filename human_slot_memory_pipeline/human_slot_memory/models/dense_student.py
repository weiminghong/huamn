from __future__ import annotations

import torch
from torch import nn

from .dinov3_backbone import DinoV3Backbone
from .heads import AttributeHead, BoxHead, PlaceholderRelationHead, PoseHead, SegmentationHead
from .slot_router import HumanSlotRouter


class DenseHumanStudent(nn.Module):
    def __init__(
        self,
        dinov3_model_path: str,
        hidden_dim: int = 256,
        max_queries: int = 32,
        freeze_dinov3: bool = True,
        dry_run: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = DinoV3Backbone(
            model_path=dinov3_model_path,
            hidden_dim=hidden_dim,
            freeze=freeze_dinov3,
            dry_run=dry_run,
        )
        self.router = HumanSlotRouter(hidden_dim=hidden_dim, max_queries=max_queries)
        self.box_head = BoxHead(hidden_dim)
        self.pose_head = PoseHead(hidden_dim)
        self.attr_head = AttributeHead(hidden_dim)
        self.segmentation_head = SegmentationHead(hidden_dim)
        self.relation_head = PlaceholderRelationHead(hidden_dim)

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(pixel_values)
        routed = self.router(features.dense_features)
        queries = routed["queries"]
        outputs = {
            "queries": queries,
            "validity_logits": routed["validity_logits"],
            "box_xyxy_norm": self.box_head(queries),
            "pose17_xyv_norm": self.pose_head(queries),
            "mask_logits": self.segmentation_head(queries, features.feature_map),
            "relation_logits": self.relation_head(queries),
        }
        outputs.update(self.attr_head(queries))
        return outputs
