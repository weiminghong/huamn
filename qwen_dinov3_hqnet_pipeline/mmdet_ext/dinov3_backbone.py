from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mmdet.models.builder import BACKBONES
except Exception:
    BACKBONES = None


def _register_backbone(cls):
    if BACKBONES is None:
        return cls
    return BACKBONES.register_module()(cls)


def _select_hidden_states(hidden_states: Sequence[torch.Tensor], out_indices: Sequence[int]) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    total = len(hidden_states)
    for index in out_indices:
        idx = int(index)
        if idx < 0:
            idx = total + idx
        idx = min(max(0, idx), total - 1)
        selected.append(hidden_states[idx])
    return selected


def _tokens_to_map(tokens: torch.Tensor, image_hw: Tuple[int, int], patch_size: int) -> torch.Tensor:
    if tokens.ndim == 4:
        return tokens
    batch, num_tokens, channels = tokens.shape
    if int((num_tokens - 1) ** 0.5) ** 2 == num_tokens - 1:
        tokens = tokens[:, 1:, :]
        num_tokens = tokens.shape[1]
    height = max(1, image_hw[0] // patch_size)
    width = max(1, image_hw[1] // patch_size)
    if height * width != num_tokens:
        side = int(num_tokens**0.5)
        height = side
        width = max(1, num_tokens // max(1, side))
    usable = min(num_tokens, height * width)
    tokens = tokens[:, :usable, :]
    return tokens.transpose(1, 2).reshape(batch, channels, height, width)


@_register_backbone
class DINOv3PyramidBackbone(nn.Module):
    """HF DINOv3 wrapper that emits multi-scale feature maps for HumanQueryNet."""

    def __init__(
        self,
        model_name_or_path: str,
        out_indices: Sequence[int] = (-4, -3, -2, -1),
        out_channels: int = 256,
        num_outs: int = 4,
        patch_size: int = 16,
        freeze_backbone: bool = False,
        trust_remote_code: bool = True,
        init_cfg: Any = None,
    ) -> None:
        super().__init__()
        from transformers import AutoModel

        self.model_name_or_path = str(model_name_or_path)
        self.out_indices = tuple(int(x) for x in out_indices)
        self.out_channels = int(out_channels)
        self.num_outs = int(num_outs)
        self.patch_size = int(patch_size)
        self.freeze_backbone = bool(freeze_backbone)
        self.init_cfg = init_cfg

        self.backbone = AutoModel.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        if hasattr(self.backbone, "config"):
            cfg_patch = getattr(self.backbone.config, "patch_size", None)
            if isinstance(cfg_patch, int) and cfg_patch > 0:
                self.patch_size = int(cfg_patch)

        self.proj_layers = nn.ModuleList(
            [nn.LazyConv2d(self.out_channels, kernel_size=1) for _ in range(max(1, self.num_outs))]
        )
        if self.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def init_weights(self) -> None:
        return None

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _forward_backbone(self, x: torch.Tensor):
        outputs = self.backbone(
            pixel_values=x,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            last_hidden = getattr(outputs, "last_hidden_state", None)
            if last_hidden is None:
                raise RuntimeError("DINOv3 backbone did not return hidden states.")
            hidden_states = [last_hidden]
        return list(hidden_states)

    def forward(self, x: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                hidden_states = self._forward_backbone(x)
        else:
            hidden_states = self._forward_backbone(x)

        selected = _select_hidden_states(hidden_states, self.out_indices)
        feats: List[torch.Tensor] = []
        for idx, hidden in enumerate(selected[: self.num_outs]):
            fmap = _tokens_to_map(hidden, image_hw=tuple(x.shape[-2:]), patch_size=self.patch_size)
            feats.append(self.proj_layers[idx](fmap))

        if not feats:
            raise RuntimeError("No DINOv3 feature maps were produced.")

        while len(feats) < self.num_outs:
            pooled = F.avg_pool2d(feats[-1], kernel_size=2, stride=2, ceil_mode=True)
            feats.append(self.proj_layers[len(feats)](pooled) if len(feats) < len(self.proj_layers) else pooled)
        return tuple(feats[: self.num_outs])
