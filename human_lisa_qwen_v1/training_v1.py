from __future__ import annotations

import math
import random
from pathlib import Path
import re
from types import MethodType
from typing import Any, Dict, Iterable, Tuple

import numpy as np
import torch


TASK_PREFIX_TO_FAMILY = {
    "box_head.": "box",
    "pose_decoder.": "pose",
    "mask_decoder.": "mask",
}
OPTIMIZER_FAMILIES = ("visual", "language", "box", "pose", "mask")


def _is_no_decay(name: str, parameter: torch.Tensor) -> bool:
    lower = name.lower()
    return (
        parameter.ndim < 2
        or name.endswith(".bias")
        or "norm" in lower
        or "embedding" in lower
    )



def _parameter_family(name: str) -> str:
    for prefix, family in TASK_PREFIX_TO_FAMILY.items():
        if name.startswith(prefix):
            return family
    if ".visual." in name:
        return "visual"
    return "language"


def set_trainable_families(
    model: torch.nn.Module,
    families: set[str],
) -> Dict[str, int]:
    selected = {str(family) for family in families}
    invalid = selected - set(OPTIMIZER_FAMILIES)
    if not selected or invalid:
        raise ValueError(
            f"trainable families must be a non-empty subset of "
            f"{OPTIMIZER_FAMILIES}: invalid={sorted(invalid)}"
        )
    counts = {family: 0 for family in OPTIMIZER_FAMILIES}
    for name, parameter in model.named_parameters():
        family = _parameter_family(name)
        if family not in selected:
            parameter.requires_grad_(False)
        if parameter.requires_grad:
            counts[family] += int(parameter.numel())
    missing = selected - {name for name, count in counts.items() if count > 0}
    if missing:
        raise RuntimeError(
            f"selected trainable families have no parameters: {sorted(missing)}"
        )
    return counts


def restrict_visual_lora_to_blocks(
    model: torch.nn.Module, start_block: int
) -> Dict[str, int]:
    """Keep visual LoRA trainable only from the requested transformer block."""
    if start_block < 0:
        raise ValueError("visual LoRA start block must be non-negative")
    frozen = 0
    kept = 0
    for name, parameter in model.named_parameters():
        match = re.search(r"(?:^|\.)visual\.blocks\.(\d+)\.", name)
        if match is None or not parameter.requires_grad:
            continue
        if int(match.group(1)) < start_block:
            parameter.requires_grad_(False)
            frozen += int(parameter.numel())
        else:
            kept += int(parameter.numel())
    if kept <= 0:
        raise RuntimeError("visual LoRA block restriction froze every visual LoRA parameter")
    return {"start_block": int(start_block), "trainable": kept, "frozen": frozen}


def prepare_mixed_precision_parameters(
    model: torch.nn.Module, frozen_dtype: torch.dtype
) -> Dict[str, int]:
    if frozen_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("frozen dtype must be float16 or bfloat16")
    from peft import cast_mixed_precision_params

    cast_mixed_precision_params(model, frozen_dtype)
    summary: Dict[str, int] = {}
    for parameter in model.parameters():
        key = "trainable_fp32" if parameter.requires_grad else "frozen_low_precision"
        summary[key] = summary.get(key, 0) + int(parameter.numel())
    return summary


def force_visual_lora_fp32_updates(model: torch.nn.Module) -> int:
    """Run only visual PEFT LoRA updates in FP32 under a BF16 outer autocast."""
    try:
        from peft.tuners.lora.layer import Linear as PeftLoraLinear
    except ImportError as error:
        raise RuntimeError("visual LoRA FP32 updates require PEFT") from error

    def fp32_lora_forward(self: torch.nn.Module, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            return self.base_layer(x, *args, **kwargs)
        if adapter_names is not None:
            return self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
        if self.merged:
            return self.base_layer(x, *args, **kwargs)
        result = self.base_layer(x, *args, **kwargs)
        result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A:
                continue
            lora_a = self.lora_A[active_adapter]
            lora_b = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            if self.use_dora[active_adapter]:
                raise RuntimeError("DoRA is unsupported for FP32 visual LoRA updates")
            with torch.autocast(x.device.type, enabled=False):
                update = lora_b(lora_a(dropout(x.float()))) * scaling
            result = result + update.to(result_dtype)
        return result.to(result_dtype)

    patched = 0
    for name, module in model.named_modules():
        if ".visual." not in f".{name}." or not isinstance(module, PeftLoraLinear):
            continue
        if getattr(module, "_stage1_fp32_visual_lora", False):
            continue
        if not any(parameter.requires_grad for parameter in module.parameters()):
            continue
        module.forward = MethodType(fp32_lora_forward, module)
        module._stage1_fp32_visual_lora = True
        patched += 1
    if patched <= 0:
        raise RuntimeError("no visual PEFT LoRA linear modules were patched")
    return patched


def build_optimizer(
    model: torch.nn.Module,
    *,
    base_lr: float,
    head_lr: float,
    weight_decay: float,
    visual_lr: float | None = None,
    language_lr: float | None = None,
    box_lr: float | None = None,
    pose_lr: float | None = None,
    mask_lr: float | None = None,
) -> Tuple[torch.optim.Optimizer, Dict[str, int]]:
    rates = {
        "visual": float(base_lr if visual_lr is None else visual_lr),
        "language": float(base_lr if language_lr is None else language_lr),
        "box": float(head_lr if box_lr is None else box_lr),
        "pose": float(head_lr if pose_lr is None else pose_lr),
        "mask": float(head_lr if mask_lr is None else mask_lr),
    }
    groups: Dict[Tuple[str, bool], list] = {
        (family, no_decay): []
        for family in OPTIMIZER_FAMILIES
        for no_decay in (False, True)
    }
    counts = {
        "visual_parameters": 0,
        "language_parameters": 0,
        "head_parameters": 0,
    }
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        family = _parameter_family(name)
        no_decay = _is_no_decay(name, parameter)
        groups[(family, no_decay)].append(parameter)
        count_family = family if family in ("visual", "language") else "head"
        counts[f"{count_family}_parameters"] += int(parameter.numel())
    if not seen:
        raise RuntimeError("model has no trainable parameters")

    optimizer_groups = []
    for family in OPTIMIZER_FAMILIES:
        for no_decay in (False, True):
            parameters = groups[(family, no_decay)]
            if not parameters:
                continue
            optimizer_groups.append(
                {
                    "params": parameters,
                    "lr": rates[family],
                    "weight_decay": 0.0 if no_decay else float(weight_decay),
                    "group_name": (
                        f"{family}_{'no_decay' if no_decay else 'decay'}"
                    ),
                }
            )
    optimizer = torch.optim.AdamW(optimizer_groups)
    counts["base_parameters"] = (
        counts["visual_parameters"] + counts["language_parameters"]
    )
    counts["trainable_parameters"] = (
        counts["base_parameters"] + counts["head_parameters"]
    )
    counts["optimizer_groups"] = len(optimizer_groups)
    return optimizer, counts


def clip_gradient_families(
    model: torch.nn.Module, *, max_norm: float
) -> Dict[str, float]:
    max_norm = float(max_norm)
    if max_norm <= 0:
        raise ValueError("max_norm must be positive")
    parameters: Dict[str, list] = {name: [] for name in OPTIMIZER_FAMILIES}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        parameters[_parameter_family(name)].append(parameter)
    norms: Dict[str, float] = {}
    for family, family_parameters in parameters.items():
        if not family_parameters:
            continue
        norm = torch.nn.utils.clip_grad_norm_(
            family_parameters, max_norm=max_norm
        )
        norms[family] = float(norm.detach().float().cpu())
    return norms


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_optimizer_steps: int,
    warmup_ratio: float = 0.03,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_optimizer_steps = int(total_optimizer_steps)
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = max(1, int(math.ceil(total_optimizer_steps * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        denominator = max(1, total_optimizer_steps - warmup_steps)
        progress = min(
            1.0, max(0.0, float(step - warmup_steps) / float(denominator))
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



def build_warmup_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_optimizer_steps: int,
    warmup_ratio: float = 0.03,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_optimizer_steps = int(total_optimizer_steps)
    if total_optimizer_steps <= 0:
        raise ValueError("total_optimizer_steps must be positive")
    if not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("warmup_ratio must be in [0, 1)")
    warmup_steps = max(1, int(math.ceil(total_optimizer_steps * warmup_ratio)))

    def lr_lambda(step: int) -> float:
        step = int(step)
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def should_optimizer_step(
    micro_step: int, total_micro_steps: int, gradient_accumulation_steps: int
) -> bool:
    micro_step = int(micro_step)
    total_micro_steps = int(total_micro_steps)
    accumulation = max(1, int(gradient_accumulation_steps))
    if micro_step <= 0 or total_micro_steps <= 0:
        raise ValueError("step counts must be positive")
    return micro_step % accumulation == 0 or micro_step == total_micro_steps


def accumulation_group_size(
    micro_step: int, total_micro_steps: int, gradient_accumulation_steps: int
) -> int:
    accumulation = max(1, int(gradient_accumulation_steps))
    if micro_step <= 0 or micro_step > total_micro_steps:
        raise ValueError("micro_step is outside the training plan")
    group_start = ((int(micro_step) - 1) // accumulation) * accumulation + 1
    return min(accumulation, int(total_micro_steps) - group_start + 1)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def install_visual_gradient_sanitizer(model: torch.nn.Module) -> Any:
    """Zero non-finite visual-LoRA gradients and expose an auditable event counter."""
    state: Dict[str, Any] = {"events": 0, "parameters": set(), "hooks": 0}

    def hook_for(name: str):
        def sanitize(gradient: torch.Tensor) -> torch.Tensor:
            if torch.isfinite(gradient).all():
                return gradient
            state["events"] += 1
            state["parameters"].add(name)
            return torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        return sanitize

    for name, parameter in model.named_parameters():
        if parameter.requires_grad and re.search(r"(?:^|\.)visual\.blocks\.", name):
            parameter.register_hook(hook_for(name))
            state["hooks"] += 1
    if state["hooks"] <= 0:
        raise RuntimeError("no trainable visual LoRA parameters received gradient sanitizers")

    def drain() -> Dict[str, Any]:
        result = {
            "events": int(state["events"]),
            "parameters": sorted(state["parameters"]),
        }
        state["events"] = 0
        state["parameters"].clear()
        return result

    return {"hooks": int(state["hooks"]), "drain": drain}


def assert_finite_training_state(
    loss: torch.Tensor, model: torch.nn.Module
) -> None:
    if not torch.isfinite(loss.detach()).all():
        raise FloatingPointError(f"non-finite loss: {loss.detach()}")
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def assert_finite_optimizer_state(
    optimizer: torch.optim.Optimizer, model: torch.nn.Module
) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError("non-finite parameter after step: " + name)
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError(
                    "non-finite optimizer state after step: "
                    + name + "." + state_name
                )




def _rng_state() -> Dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and "torch_cuda" in state:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: Dict[str, Any],
    include_model_state: bool = True,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "train_state": dict(train_state),
        "rng_state": _rng_state(),
    }
    if include_model_state:
        payload["model"] = model.state_dict()
    torch.save(payload, path)


def load_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    restore_rng: bool = True,
    load_model_state: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if load_model_state:
        if "model" not in payload:
            raise RuntimeError("checkpoint does not contain model state")
        model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    if restore_rng:
        _restore_rng_state(payload["rng_state"])
    return dict(payload["train_state"])

