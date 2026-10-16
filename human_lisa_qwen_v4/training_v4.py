from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Tuple

import torch
import torch.nn as nn


TRAINABLE_FAMILIES = (
    "visual_lora",
    "merger",
    "token_rows",
    "pose_decoder",
    "mask_decoder",
)


class EpochRandomSampler:
    """Deterministic epoch shuffle with an exact sample-offset resume point."""

    def __init__(self, data_source, *, seed: int) -> None:
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0
        self.start_index = 0

    def set_epoch(self, epoch: int, *, start_index: int = 0) -> None:
        size = len(self.data_source)
        if start_index < 0 or start_index > size:
            raise ValueError(
                f"sampler start_index must be in [0, {size}], got {start_index}"
            )
        self.epoch = int(epoch)
        self.start_index = int(start_index)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(
            len(self.data_source), generator=generator
        ).tolist()
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return len(self.data_source) - self.start_index


def discover_visual_lora_targets(
    model: nn.Module, *, start_block: int
) -> List[str]:
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        padded = f".{name}."
        if ".visual." not in padded or ".merger." in padded:
            continue
        match = re.search(r"\.visual\.blocks\.(\d+)\.", padded)
        if match is None or int(match.group(1)) < int(start_block):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError(
            f"no visual linear modules found from block {start_block}"
        )
    return sorted(targets)


def apply_visual_lora(
    model: nn.Module,
    *,
    start_block: int,
    rank: int,
    alpha: int,
    dropout: float,
) -> Tuple[nn.Module, List[str]]:
    from peft import LoraConfig, get_peft_model

    targets = discover_visual_lora_targets(model, start_block=start_block)
    config = LoraConfig(
        r=int(rank),
        lora_alpha=int(alpha),
        lora_dropout=float(dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=targets,
        modules_to_save=[],
    )
    wrapped = get_peft_model(model, config)
    return wrapped, targets


def unfreeze_visual_merger(model: nn.Module) -> Dict[str, int]:
    names = []
    count = 0
    for name, parameter in model.named_parameters():
        padded = f".{name}."
        if ".visual.merger." not in padded:
            continue
        parameter.requires_grad_(True)
        names.append(name)
        count += int(parameter.numel())
    if count <= 0:
        raise RuntimeError("no native visual merger parameters were found")
    return {"parameters": count, "tensors": len(names), "examples": names[:8]}


def parameter_family(name: str) -> str:
    if name.startswith("token_rows."):
        return "token_rows"
    if name.startswith("pose_decoder."):
        return "pose_decoder"
    if name.startswith("mask_decoder."):
        return "mask_decoder"
    padded = f".{name}."
    if "lora_" in name and ".visual." in padded:
        return "visual_lora"
    if ".visual.merger." in padded:
        return "merger"
    return "unexpected"


def trainable_manifest(model: nn.Module) -> Dict[str, Any]:
    counts = {name: 0 for name in TRAINABLE_FAMILIES}
    examples: Dict[str, List[str]] = {name: [] for name in TRAINABLE_FAMILIES}
    unexpected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_family(name)
        if family == "unexpected":
            unexpected.append((name, int(parameter.numel())))
            continue
        counts[family] += int(parameter.numel())
        if len(examples[family]) < 8:
            examples[family].append(name)
    if unexpected:
        raise RuntimeError(
            "unexpected trainable parameters, including possible LLM weights: "
            f"{unexpected[:12]}"
        )
    if counts["token_rows"] <= 0:
        raise RuntimeError("selective token rows are not trainable")
    if counts["pose_decoder"] <= 0 or counts["mask_decoder"] <= 0:
        raise RuntimeError("dense decoders are not trainable")
    return {
        "trainable_total": sum(counts.values()),
        "trainable_by_family": counts,
        "examples": examples,
    }


def _no_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    lower = name.lower()
    return (
        parameter.ndim < 2
        or name.endswith(".bias")
        or "norm" in lower
        or "embedding" in lower
    )


def build_optimizer(
    model: nn.Module,
    *,
    visual_lr: float,
    merger_lr: float,
    token_lr: float,
    head_lr: float,
    weight_decay: float,
) -> Tuple[torch.optim.Optimizer, Dict[str, Any]]:
    rates = {
        "visual_lora": float(visual_lr),
        "merger": float(merger_lr),
        "token_rows": float(token_lr),
        "pose_decoder": float(head_lr),
        "mask_decoder": float(head_lr),
    }
    grouped: Dict[Tuple[str, bool], List[torch.Tensor]] = {}
    counts = {name: 0 for name in TRAINABLE_FAMILIES}
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        family = parameter_family(name)
        if family == "unexpected":
            raise RuntimeError(f"cannot optimize unexpected parameter: {name}")
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate optimizer parameter: {name}")
        seen.add(id(parameter))
        no_decay = _no_weight_decay(name, parameter)
        grouped.setdefault((family, no_decay), []).append(parameter)
        counts[family] += int(parameter.numel())
    optimizer_groups = []
    for (family, no_decay), parameters in grouped.items():
        optimizer_groups.append(
            {
                "params": parameters,
                "lr": rates[family],
                "initial_lr": rates[family],
                "weight_decay": 0.0 if no_decay else float(weight_decay),
                "group_name": family + ("_no_decay" if no_decay else "_decay"),
                "family": family,
            }
        )
    optimizer = torch.optim.AdamW(
        optimizer_groups, betas=(0.9, 0.999), eps=1e-8
    )
    return optimizer, {
        "trainable_by_family": counts,
        "groups": [
            {
                "name": group["group_name"],
                "family": group["family"],
                "lr": group["initial_lr"],
                "weight_decay": group["weight_decay"],
            }
            for group in optimizer_groups
        ],
    }


def scheduled_route_loss_weights(
    *,
    optimizer_step: int,
    start_weight: float,
    final_pose_weight: float,
    final_seg_weight: float,
    hold_steps: int,
    warmup_steps: int,
) -> Dict[str, float]:
    values = {
        "start_weight": float(start_weight),
        "final_pose_weight": float(final_pose_weight),
        "final_seg_weight": float(final_seg_weight),
    }
    if any(value < 0 for value in values.values()):
        raise ValueError(f"route loss weights must be non-negative: {values}")
    if int(hold_steps) < 0 or int(warmup_steps) < 0:
        raise ValueError("route loss schedule steps must be non-negative")
    step = max(0, int(optimizer_step))
    if step < int(hold_steps):
        progress = 0.0
    elif int(warmup_steps) == 0:
        progress = 1.0
    else:
        progress = min(
            1.0,
            float(step - int(hold_steps) + 1) / float(warmup_steps),
        )
    return {
        "pose_route": values["start_weight"]
        + (values["final_pose_weight"] - values["start_weight"]) * progress,
        "seg_route": values["start_weight"]
        + (values["final_seg_weight"] - values["start_weight"]) * progress,
    }


def set_step_learning_rates(
    optimizer: torch.optim.Optimizer,
    *,
    optimizer_step: int,
    total_steps: int,
    warmup_steps: int,
    decoder_warmup_steps: int,
    visual_warmup_steps: int,
) -> Dict[str, float]:
    step = max(0, int(optimizer_step))
    if step < warmup_steps:
        schedule_scale = float(step + 1) / max(1, warmup_steps)
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        schedule_scale = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    rates = {}
    for group in optimizer.param_groups:
        family = str(group["family"])
        if family in {"visual_lora", "merger"}:
            if step < int(decoder_warmup_steps):
                family_scale = 0.0
            elif int(visual_warmup_steps) == 0:
                family_scale = 1.0
            else:
                family_scale = min(
                    1.0,
                    float(step - int(decoder_warmup_steps) + 1)
                    / float(visual_warmup_steps),
                )
        else:
            family_scale = 1.0
        rate = float(group["initial_lr"]) * schedule_scale * family_scale
        group["lr"] = rate
        rates[str(group["group_name"])] = rate
    return rates


def suppress_zero_lr_gradients(
    optimizer: torch.optim.Optimizer,
) -> Dict[str, Any]:
    """Make zero-LR groups truly frozen for clipping and optimizer state."""
    suppressed_by_family: Dict[str, Dict[str, int]] = {}
    for group in optimizer.param_groups:
        if float(group["lr"]) != 0.0:
            continue
        family = str(group["family"])
        stats = suppressed_by_family.setdefault(
            family, {"parameters": 0, "elements": 0}
        )
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            stats["parameters"] += 1
            stats["elements"] += int(parameter.numel())
            parameter.grad = None
    return {
        "policy": "set_to_none",
        "families": suppressed_by_family,
        "parameters": sum(
            stats["parameters"] for stats in suppressed_by_family.values()
        ),
        "elements": sum(
            stats["elements"] for stats in suppressed_by_family.values()
        ),
    }


def gradient_norms_by_family(model: nn.Module) -> Dict[str, float]:
    squared = {name: 0.0 for name in TRAINABLE_FAMILIES}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        family = parameter_family(name)
        if family == "unexpected":
            raise RuntimeError(f"unexpected gradient parameter: {name}")
        gradient = parameter.grad.detach()
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite gradient in {name}")
        squared[family] += float(gradient.float().square().sum().item())
    return {name: math.sqrt(value) for name, value in squared.items()}


def clip_gradients_by_group(
    model: nn.Module,
    *,
    token_rows_max_norm: float,
    non_token_max_norm: float,
) -> Dict[str, Dict[str, float]]:
    """Clip routing-token gradients without shrinking dense/visual updates."""
    limits = {
        "token_rows": float(token_rows_max_norm),
        "non_token": float(non_token_max_norm),
    }
    if any(value <= 0 for value in limits.values()):
        raise ValueError(f"gradient clip norms must be positive: {limits}")

    grouped: Dict[str, List[torch.Tensor]] = {
        "token_rows": [],
        "non_token": [],
    }
    seen = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.grad is None:
            continue
        family = parameter_family(name)
        if family == "unexpected":
            raise RuntimeError(f"unexpected gradient parameter: {name}")
        if id(parameter) in seen:
            raise RuntimeError(f"duplicate gradient parameter: {name}")
        seen.add(id(parameter))
        group = "token_rows" if family == "token_rows" else "non_token"
        grouped[group].append(parameter)

    missing = [name for name, parameters in grouped.items() if not parameters]
    if missing:
        raise RuntimeError(f"gradient clip groups have no gradients: {missing}")

    pre_clip = {}
    for name, parameters in grouped.items():
        value = torch.nn.utils.clip_grad_norm_(
            parameters,
            max_norm=limits[name],
            error_if_nonfinite=True,
        )
        pre_clip[name] = float(value.detach().float().cpu().item())

    family_post = gradient_norms_by_family(model)
    post_clip = {
        "token_rows": family_post["token_rows"],
        "non_token": math.sqrt(
            sum(
                value * value
                for family, value in family_post.items()
                if family != "token_rows"
            )
        ),
    }
    scales = {
        name: min(1.0, limits[name] / max(pre_clip[name], 1e-12))
        for name in limits
    }
    return {
        "max_norms": limits,
        "pre_clip_norms": pre_clip,
        "post_clip_norms": post_clip,
        "clip_scales": scales,
    }


def assert_finite_parameters(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and not torch.isfinite(parameter).all():
            raise FloatingPointError(f"non-finite trainable parameter: {name}")


def assert_finite_optimizer(optimizer: torch.optim.Optimizer) -> None:
    for parameter, state in optimizer.state.items():
        del parameter
        for name, value in state.items():
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"non-finite optimizer state: {name}")


def move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {name: move_to_device(item, device) for name, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def trainable_merger_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter_family(name) == "merger" and parameter.requires_grad
    }
