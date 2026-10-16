from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple
import math

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
try:
    from mmdet.registry import MODELS
except Exception:
    MODELS = None


def _register_backbone(cls):
    registered = cls
    if BACKBONES is not None:
        registered = BACKBONES.register_module(force=True)(registered)
    if MODELS is not None and MODELS is not BACKBONES:
        registered = MODELS.register_module(force=True)(registered)
    return registered


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
        target_strides: Sequence[int] = (8, 16, 32, 64),
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
        self.target_strides = tuple(int(s) for s in target_strides)
        if len(self.target_strides) < self.num_outs:
            last_stride = self.target_strides[-1] if self.target_strides else self.patch_size
            self.target_strides = self.target_strides + tuple(last_stride * (2 ** i) for i in range(1, self.num_outs - len(self.target_strides) + 1))
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

    def _hidden_states_to_pyramid(self, hidden_states: Sequence[torch.Tensor], image_hw: Tuple[int, int]):
        selected = _select_hidden_states(hidden_states, self.out_indices)
        if not selected:
            raise RuntimeError("No DINOv3 hidden states were selected.")

        image_h, image_w = int(image_hw[0]), int(image_hw[1])
        feats: List[torch.Tensor] = []
        for idx in range(self.num_outs):
            hidden = selected[min(idx, len(selected) - 1)]
            fmap = _tokens_to_map(hidden, image_hw=image_hw, patch_size=self.patch_size)
            stride = self.target_strides[idx]
            target_h = max(1, int(math.ceil(image_h / float(stride))))
            target_w = max(1, int(math.ceil(image_w / float(stride))))
            if tuple(fmap.shape[-2:]) != (target_h, target_w):
                fmap = F.interpolate(fmap, size=(target_h, target_w), mode="bilinear", align_corners=False)
            feats.append(self.proj_layers[idx](fmap))
        return tuple(feats)

    def forward(self, x: torch.Tensor):
        if self.freeze_backbone:
            with torch.no_grad():
                hidden_states = self._forward_backbone(x)
        else:
            hidden_states = self._forward_backbone(x)
        return self._hidden_states_to_pyramid(hidden_states, image_hw=tuple(x.shape[-2:]))


@_register_backbone
class FrozenQwenDINOv3Backbone(DINOv3PyramidBackbone):
    """Online Stage2 backbone: frozen Qwen image encoder -> DINOv3 -> HQNet.

    fusion_mode controls how the frozen Qwen image encoder interacts with DINOv3:
    - feature_add: project Qwen features to each DINO pyramid level and add.
    - feature_concat: concatenate Qwen and DINO features per level, then reduce.
    - input_add: legacy pixel-space bridge added to DINO input.
    - input_replace: legacy pixel-space bridge used as DINO input.
    - none: load Qwen but return the plain DINO pyramid for ablation checks.
    Legacy names add/replace are accepted as aliases for input_add/input_replace.
    """

    _FUSION_ALIASES = {
        "add": "input_add",
        "replace": "input_replace",
    }
    _SUPPORTED_FUSION_MODES = {"feature_add", "feature_concat", "input_add", "input_replace", "none"}

    def __init__(
        self,
        qwen_model_name_or_path: str,
        qwen_adapter_path: str = "",
        qwen_torch_dtype: str = "bfloat16",
        qwen_merge_lora: bool = True,
        qwen_input_mean: Sequence[float] = (123.675, 116.28, 103.53),
        qwen_input_std: Sequence[float] = (58.395, 57.12, 57.375),
        bridge_hidden_channels: int = 64,
        fusion_mode: str = "feature_add",
        bridge_scale: float = 1.0,
        qwen_image_mean: Optional[Sequence[float]] = None,
        qwen_image_std: Optional[Sequence[float]] = None,
        qwen_rescale_factor: Optional[float] = None,
        freeze_qwen: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        requested_fusion_mode = str(fusion_mode)
        self.fusion_mode = self._FUSION_ALIASES.get(requested_fusion_mode, requested_fusion_mode)
        if self.fusion_mode not in self._SUPPORTED_FUSION_MODES:
            allowed = sorted(self._SUPPORTED_FUSION_MODES | set(self._FUSION_ALIASES))
            raise ValueError(f"Unsupported fusion_mode={requested_fusion_mode}. Allowed: {allowed}")
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
        self.qwen_image_processor = getattr(qwen_processor, "image_processor", None)
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
        self.qwen_feature_projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyConv2d(int(bridge_hidden_channels), kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(int(bridge_hidden_channels), self.out_channels, kernel_size=1),
                )
                for _ in range(max(1, self.num_outs))
            ]
        )
        self.feature_fuse_layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.out_channels * 2, self.out_channels, kernel_size=1),
                    nn.GELU(),
                    nn.Conv2d(self.out_channels, self.out_channels, kernel_size=1),
                )
                for _ in range(max(1, self.num_outs))
            ]
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

    def _denormalize_to_processor_images(self, x: torch.Tensor) -> List[torch.Tensor]:
        pixels = x.detach().float() * self.qwen_input_std + self.qwen_input_mean
        pixels = pixels.clamp(0, 255).to(dtype=torch.uint8).cpu()
        return [img for img in pixels]

    def _qwen_spatial_merge_size(self) -> int:
        for obj in (self.qwen_vision, getattr(self.qwen_vision, "config", None)):
            value = getattr(obj, "spatial_merge_size", None) if obj is not None else None
            if isinstance(value, int) and value > 0:
                return int(value)
        return 1

    def _qwen_tokens_from_grid_to_feature_map(
        self,
        qwen_tokens: torch.Tensor,
        image_grid_thw: Optional[torch.Tensor],
        batch_size: int,
    ) -> torch.Tensor:
        if qwen_tokens.ndim == 3 and qwen_tokens.shape[0] == batch_size:
            return token_sequence_to_feature_map(qwen_tokens, image_size=None, patch_size=self.qwen_patch_size)
        if qwen_tokens.ndim != 2 or image_grid_thw is None:
            if qwen_tokens.ndim == 2:
                qwen_tokens = qwen_tokens.unsqueeze(0)
            return token_sequence_to_feature_map(qwen_tokens, image_size=None, patch_size=self.qwen_patch_size)

        merge = self._qwen_spatial_merge_size()
        maps: List[torch.Tensor] = []
        offset = 0
        for row in image_grid_thw.detach().cpu().tolist():
            grid_t, grid_h, grid_w = [max(1, int(v)) for v in row[:3]]
            feat_h = max(1, grid_h // merge)
            feat_w = max(1, grid_w // merge)
            count = grid_t * feat_h * feat_w
            if offset + count > qwen_tokens.shape[0]:
                raise RuntimeError(
                    f"Qwen token/grid mismatch: need {offset + count}, got {qwen_tokens.shape[0]}, "
                    f"grid={(grid_t, grid_h, grid_w)}, merge={merge}"
                )
            chunk = qwen_tokens[offset: offset + count]
            offset += count
            if grid_t > 1:
                chunk = chunk.reshape(grid_t, feat_h * feat_w, -1).mean(dim=0)
            chunk = chunk[: feat_h * feat_w]
            fmap = chunk.transpose(0, 1).reshape(chunk.shape[-1], feat_h, feat_w)
            maps.append(fmap)
        if not maps:
            raise RuntimeError("No Qwen feature maps were built from image_grid_thw.")
        max_h = max(m.shape[-2] for m in maps)
        max_w = max(m.shape[-1] for m in maps)
        aligned = [
            F.interpolate(m.unsqueeze(0), size=(max_h, max_w), mode="bilinear", align_corners=False).squeeze(0)
            if tuple(m.shape[-2:]) != (max_h, max_w) else m
            for m in maps
        ]
        return torch.stack(aligned, dim=0)

    def _extract_qwen_feature_map_with_processor(self, x: torch.Tensor) -> torch.Tensor:
        if self.qwen_image_processor is None:
            raise RuntimeError("Qwen image processor is unavailable for patchified vision input fallback.")
        images = self._denormalize_to_processor_images(x)
        batch = self.qwen_image_processor(images=images, return_tensors="pt")
        pixel_values = batch["pixel_values"].to(device=x.device, dtype=next(self.qwen_vision.parameters()).dtype)
        image_grid_thw = batch.get("image_grid_thw", None)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device=x.device)
        with torch.no_grad():
            qwen_tokens = extract_qwen_vision_features_from_module(
                self.qwen_vision,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        return self._qwen_tokens_from_grid_to_feature_map(qwen_tokens, image_grid_thw, batch_size=x.shape[0])

    def _extract_qwen_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        if self.qwen_image_processor is not None:
            return self._extract_qwen_feature_map_with_processor(x)
        pixel_values = self._prepare_qwen_pixels(x)
        with torch.no_grad():
            qwen_tokens = extract_qwen_vision_features_from_module(self.qwen_vision, pixel_values=pixel_values)
        if qwen_tokens.ndim == 2:
            qwen_tokens = qwen_tokens.unsqueeze(0)
        return token_sequence_to_feature_map(
            qwen_tokens,
            image_size=tuple(pixel_values.shape[-2:]),
            patch_size=self.qwen_patch_size,
        )

    def _build_dino_input(self, x: torch.Tensor, qwen_feature_map: Optional[torch.Tensor] = None) -> torch.Tensor:
        if qwen_feature_map is None:
            qwen_feature_map = self._extract_qwen_feature_map(x)
        bridge = self.qwen_bridge(qwen_feature_map.float()).to(dtype=x.dtype)
        bridge = F.interpolate(bridge, size=tuple(x.shape[-2:]), mode="bilinear", align_corners=False)
        bridge = torch.tanh(bridge)
        if self.fusion_mode == "input_replace":
            return self.bridge_scale.to(dtype=x.dtype) * bridge
        return x + self.bridge_scale.to(dtype=x.dtype) * bridge

    def _fuse_feature_maps(
        self,
        dino_feats: Sequence[torch.Tensor],
        qwen_feature_map: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        fused_feats: List[torch.Tensor] = []
        scale = self.bridge_scale
        for idx, dino_feat in enumerate(dino_feats):
            qwen_level = F.interpolate(
                qwen_feature_map.float(),
                size=tuple(dino_feat.shape[-2:]),
                mode="bilinear",
                align_corners=False,
            )
            qwen_level = self.qwen_feature_projs[idx](qwen_level).to(dtype=dino_feat.dtype)
            if self.fusion_mode == "feature_add":
                fused = dino_feat + scale.to(dtype=dino_feat.dtype) * qwen_level
            elif self.fusion_mode == "feature_concat":
                concat_feat = torch.cat([dino_feat, scale.to(dtype=dino_feat.dtype) * qwen_level], dim=1).float()
                fused = self.feature_fuse_layers[idx](concat_feat).to(dtype=dino_feat.dtype)
            else:
                raise RuntimeError(f"Feature fusion called for unsupported mode={self.fusion_mode}")
            fused_feats.append(fused)
        return tuple(fused_feats)

    def forward(self, x: torch.Tensor):
        if self.fusion_mode == "none":
            return super().forward(x)

        if self.fusion_mode in {"input_add", "input_replace"}:
            qwen_feature_map = self._extract_qwen_feature_map(x)
            dino_input = self._build_dino_input(x, qwen_feature_map=qwen_feature_map)
            return super().forward(dino_input)

        qwen_feature_map = self._extract_qwen_feature_map(x)
        dino_feats = super().forward(x)
        return self._fuse_feature_maps(dino_feats=dino_feats, qwen_feature_map=qwen_feature_map)
