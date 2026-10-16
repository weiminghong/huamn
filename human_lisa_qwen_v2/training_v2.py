from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import torch


FAMILIES = ("visual_lora", "language_lora", "token_rows", "visual_neck", "pose", "mask")


def parameter_family(name: str) -> str:
    if name.startswith("visual_neck."):
        return "visual_neck"
    if name.startswith("pose_decoder."):
        return "pose"
    if name.startswith("mask_decoder."):
        return "mask"
    if "trainable_tokens" in name:
        return "token_rows"
    if "lora_" in name and ".visual." in name:
        return "visual_lora"
    return "language_lora"


def no_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    lower = name.lower()
    return parameter.ndim < 2 or name.endswith(".bias") or "norm" in lower or "token" in lower


def build_optimizer(
    model: torch.nn.Module,
    *,
    visual_lr: float,
    language_lr: float,
    token_lr: float,
    head_lr: float,
    weight_decay: float,
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    rates = {
        "visual_lora": float(visual_lr),
        "language_lora": float(language_lr),
        "token_rows": float(token_lr),
        "visual_neck": float(head_lr),
        "pose": float(head_lr),
        "mask": float(head_lr),
    }
    grouped: Dict[Tuple[str, bool], list] = {(family, decay): [] for family in FAMILIES for decay in (False, True)}
    counts = {family: 0 for family in FAMILIES}
    names = {family: [] for family in FAMILIES}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate trainable parameter: {name}")
        seen.add(id(parameter))
        family = parameter_family(name)
        counts[family] += int(parameter.numel())
        if len(names[family]) < 6:
            names[family].append(name)
        grouped[(family, not no_weight_decay(name, parameter))].append(parameter)
    missing = [family for family, count in counts.items() if count <= 0]
    if missing:
        raise RuntimeError(f"missing trainable parameter families: {missing}")
    optimizer_groups = []
    for family in FAMILIES:
        for decay in (False, True):
            parameters = grouped[(family, decay)]
            if parameters:
                optimizer_groups.append(
                    {
                        "params": parameters,
                        "lr": rates[family],
                        "weight_decay": float(weight_decay) if decay else 0.0,
                        "group_name": family + ("_decay" if decay else "_no_decay"),
                    }
                )
    optimizer = torch.optim.AdamW(optimizer_groups, betas=(0.9, 0.999), eps=1e-8)
    return optimizer, {"counts": counts, "examples": names, "rates": rates, "groups": len(optimizer_groups)}


def build_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(total_steps)
    warmup_steps = max(1, int(math.ceil(total_steps * float(warmup_ratio))))
    if total_steps <= 0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("invalid scheduler plan")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, total_steps - warmup_steps)))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)


def move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def gradient_statistics(model: torch.nn.Module) -> Dict[str, Dict[str, float]]:
    stats = {family: {"parameters": 0, "with_grad": 0, "nonzero": 0, "norm_squared": 0.0, "finite": True} for family in FAMILIES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_family(name)
        stats[family]["parameters"] += int(parameter.numel())
        if parameter.grad is None:
            continue
        stats[family]["with_grad"] += 1
        gradient = parameter.grad.detach()
        stats[family]["finite"] = bool(stats[family]["finite"] and torch.isfinite(gradient).all())
        norm = float(gradient.float().norm().cpu())
        stats[family]["norm_squared"] += norm * norm
        if norm > 0.0:
            stats[family]["nonzero"] += 1
    for values in stats.values():
        values["norm"] = math.sqrt(values.pop("norm_squared"))
    return stats


def assert_finite_gradients(model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"non-finite gradient: {name}")


def assert_finite_optimizer_state(optimizer: torch.optim.Optimizer, model: torch.nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if not torch.isfinite(parameter.detach()).all():
            raise FloatingPointError(f"non-finite parameter after optimizer step: {name}")
        for state_name, value in optimizer.state.get(parameter, {}).items():
            if isinstance(value, torch.Tensor) and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite optimizer state: {name}.{state_name}")


def clip_global_gradient(model: torch.nn.Module, max_norm: float) -> float:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    if not parameters:
        raise RuntimeError("optimizer step has no gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, float(max_norm))
    return float(norm.detach().float().cpu())


def scalar_tensors(outputs: Dict[str, Any]) -> Dict[str, float]:
    return {
        name: float(value.detach().float().cpu())
        for name, value in outputs.items()
        if isinstance(value, torch.Tensor) and value.numel() == 1
    }


def save_training_state(
    path: Path,
    *,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    train_state: Dict[str, Any],
) -> None:
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "train_state": dict(train_state),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        Path(path),
    )


def build_constant_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_steps: int,
    warmup_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_steps = int(total_steps)
    warmup_steps = max(1, int(math.ceil(total_steps * float(warmup_ratio))))
    if total_steps <= 0 or not 0.0 <= warmup_ratio < 1.0:
        raise ValueError("invalid scheduler plan")

    def schedule(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
