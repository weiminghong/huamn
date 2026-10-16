import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.training_v1 import (
    accumulation_group_size,
    assert_finite_training_state,
    build_optimizer,
    build_warmup_constant_scheduler,
    build_warmup_cosine_scheduler,
    clip_gradient_families,
    load_training_state,
    force_visual_lora_fp32_updates,
    install_visual_gradient_sanitizer,
    move_to_device,
    save_training_state,
    set_trainable_families,
    should_optimizer_step,
    restrict_visual_lora_to_blocks,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = nn.Module()
        self.base_model.lora_A = nn.Linear(4, 2, bias=False)
        self.pose_decoder = nn.Linear(4, 4)
        self.mask_decoder = nn.Linear(4, 4)
        self.box_head = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)
        self.frozen = nn.Linear(4, 4)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False

    def forward(self, value):
        return self.box_head(value)


def test_optimizer_groups_have_head_base_and_no_decay_semantics():
    model = TinyModel()
    optimizer, summary = build_optimizer(
        model, base_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    assert summary["trainable_parameters"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    assert summary["head_parameters"] > 0
    assert summary["base_parameters"] > 0
    assert {group["lr"] for group in optimizer.param_groups} == {2e-5, 1e-4}
    assert {group["weight_decay"] for group in optimizer.param_groups} == {
        0.0,
        0.01,
    }


def test_gradient_accumulation_uses_optimizer_step_semantics():
    assert not should_optimizer_step(1, 10, 4)
    assert not should_optimizer_step(3, 10, 4)
    assert should_optimizer_step(4, 10, 4)
    assert should_optimizer_step(8, 10, 4)
    assert should_optimizer_step(10, 10, 4)
    assert [accumulation_group_size(step, 10, 4) for step in range(1, 5)] == [4] * 4
    assert [accumulation_group_size(step, 10, 4) for step in range(5, 9)] == [4] * 4
    assert [accumulation_group_size(step, 10, 4) for step in range(9, 11)] == [2] * 2


def test_warmup_cosine_scheduler_is_counted_in_optimizer_steps():
    model = TinyModel()
    optimizer, _ = build_optimizer(
        model, base_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_optimizer_steps=100, warmup_ratio=0.03
    )
    factors = []
    for _ in range(100):
        optimizer.step()
        scheduler.step()
        factors.append(scheduler.get_last_lr()[0] / 2e-5)
    assert factors[0] < factors[1] <= factors[2]
    assert factors[2] == pytest.approx(1.0, rel=1e-6)
    assert factors[-1] < 0.001


def test_move_to_device_recurses_without_moving_metadata():
    batch = {
        "tensor": torch.ones(2),
        "nested": [torch.zeros(1), {"x": torch.tensor(3)}],
        "metadata": ["a", "b"],
    }
    moved = move_to_device(batch, torch.device("cpu"))
    assert moved["tensor"].device.type == "cpu"
    assert moved["nested"][1]["x"].device.type == "cpu"
    assert moved["metadata"] == ["a", "b"]


class TinyVisualModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False)])
        self.head = nn.Linear(2, 1, bias=False)


def test_visual_gradient_sanitizer_only_zeroes_nonfinite_visual_gradients():
    model = TinyVisualModule()
    sanitizer = install_visual_gradient_sanitizer(model)
    loss = (model.visual.blocks[0].weight * float("inf")).sum()
    loss.backward()
    visual_grad = model.visual.blocks[0].weight.grad
    assert torch.isfinite(visual_grad).all()
    assert torch.count_nonzero(visual_grad) == 0
    stats = sanitizer["drain"]()
    assert stats["events"] == 1
    assert stats["parameters"] == ["visual.blocks.0.weight"]


def test_nonfinite_loss_or_gradient_is_rejected():
    model = TinyModel()
    loss = model(torch.ones(1, 4)).sum()
    loss.backward()
    assert_finite_training_state(loss, model)
    with pytest.raises(FloatingPointError, match="non-finite loss"):
        assert_finite_training_state(torch.tensor(float("nan")), model)
    model.box_head.weight.grad[0, 0] = float("inf")
    with pytest.raises(FloatingPointError, match="non-finite gradient"):
        assert_finite_training_state(loss, model)


def test_training_state_round_trip(tmp_path):
    torch.manual_seed(10)
    model = TinyModel()
    optimizer, _ = build_optimizer(
        model, base_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_optimizer_steps=4, warmup_ratio=0.25
    )
    value = torch.randn(2, 4)
    model(value).sum().backward()
    optimizer.step()
    scheduler.step()
    save_training_state(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_state={"micro_step": 4, "optimizer_step": 1, "epoch": 0},
    )
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(1)
    state = load_training_state(
        tmp_path / "state.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    assert state["optimizer_step"] == 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])



def test_special_tokens_do_not_shrink_qwen_padded_vocab():
    from types import SimpleNamespace
    from human_lisa_qwen_v1.train_stage1_v1 import add_special_tokens

    class FakeTokenizer:
        additional_special_tokens = []

        def __init__(self):
            self.ids = {}
            self.length = 100

        def add_special_tokens(self, payload):
            tokens = payload["additional_special_tokens"]
            for token in tokens:
                if token not in self.ids:
                    self.ids[token] = self.length
                    self.length += 1

        def convert_tokens_to_ids(self, token):
            return self.ids[token]

        def __len__(self):
            return self.length

    class FakeModel:
        def __init__(self):
            self.embedding = SimpleNamespace(num_embeddings=128)
            self.resize_calls = []

        def get_input_embeddings(self):
            return self.embedding

        def resize_token_embeddings(self, size):
            self.resize_calls.append(size)

    tokenizer = FakeTokenizer()
    model = FakeModel()
    token_ids = add_special_tokens(SimpleNamespace(tokenizer=tokenizer), model)
    assert token_ids == {"box": 100, "pose": 101, "seg": 102}
    assert model.resize_calls == []


def test_decode_probe_writes_visual_overlays(tmp_path, monkeypatch):
    from PIL import Image

    import human_lisa_qwen_v1.inference_v1 as inference_v1
    from human_lisa_qwen_v1.train_stage1_v1 import run_decode_probe

    class FakeModel:
        training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

    results = [
        {
            "image_id": 11,
            "raw": '{"persons":[]}',
            "schema_ok": True,
            "eos_seen": True,
            "health": {"ok": True},
            "records": [],
        }
    ]

    def fake_infer(*args, **kwargs):
        return results

    monkeypatch.setattr(inference_v1, "infer_samples", fake_infer)
    summary = run_decode_probe(
        model=FakeModel(),
        processor=object(),
        samples=[
            {
                "id": 11,
                "image": Image.new("RGB", (32, 24), color="white"),
                "instances": [],
            }
        ],
        output_dir=tmp_path,
        optimizer_step=5,
        max_new_tokens=5000,
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    assert summary["overlay_files"] == ["step-5-image-11.png"]
    assert summary["pose_count_mae"] == pytest.approx(0.0)
    assert summary["seg_count_mae"] == pytest.approx(0.0)
    assert (tmp_path / "decode_probes" / "step-5-image-11.png").is_file()


def test_optimizer_supports_visual_language_and_head_learning_rates():
    model = TinyModel()
    model.base_model.visual = nn.Module()
    model.base_model.visual.lora_A = nn.Linear(4, 2, bias=False)
    optimizer, summary = build_optimizer(
        model,
        base_lr=2e-5,
        visual_lr=2e-5,
        language_lr=1e-4,
        head_lr=5e-4,
        box_lr=5e-4,
        pose_lr=1e-3,
        mask_lr=1e-4,
        weight_decay=0.01,
    )
    rates = {
        group["group_name"].split("_")[0]: group["lr"]
        for group in optimizer.param_groups
    }
    assert rates["visual"] == pytest.approx(2e-5)
    assert rates["language"] == pytest.approx(1e-4)
    assert rates["box"] == pytest.approx(5e-4)
    assert rates["pose"] == pytest.approx(1e-3)
    assert rates["mask"] == pytest.approx(1e-4)
    assert summary["visual_parameters"] > 0
    assert summary["language_parameters"] > 0
    assert summary["head_parameters"] > 0


def test_warmup_constant_scheduler_keeps_peak_rate_after_warmup():
    model = TinyModel()
    optimizer, _ = build_optimizer(
        model, base_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    scheduler = build_warmup_constant_scheduler(
        optimizer, total_optimizer_steps=10, warmup_ratio=0.2
    )
    factors = []
    for _ in range(10):
        factors.append(optimizer.param_groups[0]["lr"] / 2e-5)
        optimizer.step()
        scheduler.step()
    assert factors[0] < factors[1]
    assert factors[1:] == pytest.approx([1.0] * 9)


def test_gradient_clipping_is_independent_per_parameter_family():
    model = TinyModel()
    model.base_model.visual = nn.Module()
    model.base_model.visual.lora_A = nn.Linear(4, 2, bias=False)
    loss = (
        model.base_model.lora_A.weight.sum() * 100.0
        + model.base_model.visual.lora_A.weight.sum() * 1000.0
        + model.pose_decoder.weight.sum() * 20000.0
        + model.mask_decoder.weight.sum() * 30000.0
        + model.box_head.weight.sum() * 10000.0
    )
    loss.backward()
    norms = clip_gradient_families(model, max_norm=2.0)
    assert set(norms) == {"visual", "language", "box", "pose", "mask"}
    assert (
        norms["mask"] > norms["pose"] > norms["box"]
        > norms["visual"] > norms["language"]
    )

    by_family = {name: [] for name in norms}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        if name.startswith("box_head."):
            family = "box"
        elif name.startswith("pose_decoder."):
            family = "pose"
        elif name.startswith("mask_decoder."):
            family = "mask"
        elif ".visual." in name:
            family = "visual"
        else:
            family = "language"
        by_family[family].append(parameter.grad.flatten())
    for gradients in by_family.values():
        assert torch.cat(gradients).norm().item() <= 2.0001

def test_loss_weights_from_args_supports_pose_only_control():
    from types import SimpleNamespace

    from human_lisa_qwen_v1.train_stage1_v1 import loss_weights_from_args

    args = SimpleNamespace(
        ce_loss_weight=0.0,
        box_l1_loss_weight=0.0,
        box_giou_loss_weight=0.0,
        box_quality_loss_weight=0.0,
        pose_l1_loss_weight=5.0,
        pose_oks_loss_weight=1.5,
        pose_heatmap_loss_weight=0.0,
        pose_visibility_loss_weight=1.0,
        mask_bce_loss_weight=0.0,
        mask_dice_loss_weight=0.0,
    )
    assert loss_weights_from_args(args) == {
        "ce": 0.0,
        "box_l1": 0.0,
        "box_giou": 0.0,
        "box_quality": 0.0,
        "pose_l1": 5.0,
        "pose_oks": 1.5,
        "pose_heatmap": 0.0,
        "pose_visibility": 1.0,
        "mask_bce": 0.0,
        "mask_dice": 0.0,
    }


def test_set_trainable_families_excludes_frozen_parameters_from_optimizer():
    model = TinyModel()
    counts = set_trainable_families(model, {"pose"})
    assert counts["pose"] > 0
    assert counts["visual"] == 0
    assert counts["language"] == 0
    assert counts["box"] == 0
    assert counts["mask"] == 0
    assert all(
        parameter.requires_grad
        for parameter in model.pose_decoder.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("pose_decoder.")
    )
    optimizer, summary = build_optimizer(
        model, base_lr=2e-5, head_lr=1e-4, weight_decay=0.01
    )
    assert summary["base_parameters"] == 0
    assert summary["head_parameters"] == counts["pose"]
    assert {
        group["group_name"].split("_")[0]
        for group in optimizer.param_groups
    } == {"pose"}



def test_visual_lora_fp32_patch_is_scoped_and_has_finite_gradients():
    from peft.tuners.lora.layer import Linear as PeftLoraLinear

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = nn.Module()
            self.visual.lora = PeftLoraLinear(
                nn.Linear(4, 4, bias=False),
                adapter_name="default",
                r=2,
                lora_alpha=2,
            )
            self.language_lora = PeftLoraLinear(
                nn.Linear(4, 4, bias=False),
                adapter_name="default",
                r=2,
                lora_alpha=2,
            )

    model = Wrapper()
    assert force_visual_lora_fp32_updates(model) == 1
    assert getattr(model.visual.lora, "_stage1_fp32_visual_lora", False)
    assert not getattr(model.language_lora, "_stage1_fp32_visual_lora", False)
    value = torch.randn(3, 4, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model.visual.lora(value)
    output.float().square().mean().backward()
    assert torch.isfinite(model.visual.lora.lora_A["default"].weight.grad).all()


def test_restrict_visual_lora_to_later_blocks():
    class VisualBlocks(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2)])

    class Wrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.visual = VisualBlocks()
            self.language = nn.Linear(2, 2)

    model = Wrapper()
    result = restrict_visual_lora_to_blocks(model, 1)
    assert result == {"start_block": 1, "trainable": 12, "frozen": 6}
    assert not any(parameter.requires_grad for parameter in model.visual.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in model.visual.blocks[1].parameters())
    assert all(parameter.requires_grad for parameter in model.visual.blocks[2].parameters())
    assert all(parameter.requires_grad for parameter in model.language.parameters())
