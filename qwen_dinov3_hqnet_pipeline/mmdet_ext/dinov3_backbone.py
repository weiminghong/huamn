from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from qwen_dinov3_hqnet_pipeline.common.qwen_utils import (
    extract_patch_size,
    extract_qwen_vision_features_from_module,
    load_qwen_vl_model,
    resolve_vision_module,
    token_sequence_to_feature_map,
)

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
    return token_sequence_to_feature_map(tokens, image_size=image_hw, patch_size=patch_size)


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


@_register_backbone
class FrozenQwenDINOv3Backbone(DINOv3PyramidBackbone):
    """Online Stage2 backbone: frozen Qwen image encoder -> bridge -> DINOv3 -> HQNet."""

    def __init__(
        self,
        qwen_model_name_or_path: str,
        qwen_adapter_path: str = "",
        qwen_torch_dtype: str = "bfloat16",
        qwen_merge_lora: bool = True,
        qwen_input_mean: Sequence[float] = (123.675, 116.28, 103.53),
        qwen_input_std: Sequence[float] = (58.395, 57.12, 57.375),
        bridge_hidden_channels: int = 64,
        fusion_mode: str = "add",
        bridge_scale: float = 1.0,
        qwen_image_mean: Optional[Sequence[float]] = None,
        qwen_image_std: Optional[Sequence[float]] = None,
        qwen_rescale_factor: Optional[float] = None,
        freeze_qwen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode not in {"add", "replace"}:
            raise ValueError(f"Unsupported fusion_mode={self.fusion_mode}")
        self.freeze_qwen = bool(freeze_qwen)
        self.bridge_scale = nn.Parameter(torch.tensor(float(bridge_scale), dtype=torch.float32))

        qwen_model, qwen_processor = load_qwen_vl_model(
            model_name_or_path=qwen_model_name_or_path,
            adapter_path=qwen_adapter_path,
            torch_dtype=qwen_torch_dtype,
            device="cpu",
            merge_lora=bool(qwen_merge_lora),
        )
        qwen_vision, _ = resolve_vision_module(qwen_model)
        self.qwen_vision = qwen_vision
        self.qwen_patch_size = extract_patch_size(self.qwen_vision)
        if self.freeze_qwen:
            for param in self.qwen_vision.parameters():
                param.requires_grad = False

        processor_mean = None
        processor_std = None
        processor_rescale = None
        image_processor = getattr(qwen_processor, "image_processor", None)
        if image_processor is not None:
            processor_mean = getattr(image_processor, "image_mean", None)
            processor_std = getattr(image_processor, "image_std", None)
            processor_rescale = getattr(image_processor, "rescale_factor", None)

        if qwen_image_mean is None:
            qwen_image_mean = processor_mean or (0.5, 0.5, 0.5)
        if qwen_image_std is None:
            qwen_image_std = processor_std or (0.5, 0.5, 0.5)
        if qwen_rescale_factor is None:
            qwen_rescale_factor = float(processor_rescale) if processor_rescale is not None else (1.0 / 255.0)

        self.register_buffer("qwen_input_mean", torch.tensor(list(qwen_input_mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("qwen_input_std", torch.tensor(list(qwen_input_std), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("qwen_image_mean", torch.tensor(list(qwen_image_mean), dtype=torch.float32).view(1, 3, 1, 1))
        self.register_buffer("qwen_image_std", torch.tensor(list(qwen_image_std), dtype=torch.float32).view(1, 3, 1, 1))
        self.qwen_rescale_factor = float(qwen_rescale_factor)

        self.qwen_bridge = nn.Sequential(
            nn.LazyConv2d(int(bridge_hidden_channels), kernel_size=1),
            nn.GELU(),
            nn.Conv2d(int(bridge_hidden_channels), 3, kernel_size=1),
        )

        del qwen_model
        del qwen_processor

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_qwen:
            self.qwen_vision.eval()
        return self

    def _prepare_qwen_pixels(self, x: torch.Tensor) -> torch.Tensor:
        pixels = x.float()
        pixels = pixels * self.qwen_input_std + self.qwen_input_mean
        pixels = pixels * self.qwen_rescale_factor
        pixels = (pixels - self.qwen_image_mean) / self.qwen_image_std
        return pixels.to(device=x.device, dtype=next(self.qwen_vision.parameters()).dtype)

    def _extract_qwen_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        pixel_values = self._prepare_qwen_pixels(x)
        with torch.no_grad():
            qwen_tokens = extract_qwen_vision_features_from_module(self.qwen_vision, pixel_values=pixel_values)
        return token_sequence_to_feature_map(
            qwen_tokens,
            image_size=tuple(pixel_values.shape[-2:]),
            patch_size=self.qwen_patch_size,
        )

    def _build_dino_input(self, x: torch.Tensor) -> torch.Tensor:
        qwen_feature_map = self._extract_qwen_feature_map(x)
        bridge = self.qwen_bridge(qwen_feature_map.to(dtype=x.dtype))
        bridge = F.interpolate(bridge, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
        bridge = torch.tanh(bridge)
        if self.fusion_mode == "replace":
            return self.bridge_scale.to(dtype=x.dtype) * bridge
        return x + self.bridge_scale.to(dtype=x.dtype) * bridge

    def forward(self, x: torch.Tensor):
        dino_input = self._build_dino_input(x)
        return super().forward(dino_input)
