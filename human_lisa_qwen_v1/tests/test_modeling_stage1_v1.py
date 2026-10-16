import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.modeling_stage1_v1 import (
    Stage1V1Model,
    scale_task_gradient,
    select_pose_reference_boxes,
    weighted_causal_language_loss,
)


class FakeVLModel(nn.Module):
    def __init__(self, hidden_size=32, vocab_size=64):
        super().__init__()
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            text_config=SimpleNamespace(hidden_size=hidden_size),
            image_token_id=50,
            use_cache=False,
        )
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        output_hidden_states=True,
        return_dict=True,
        **kwargs,
    ):
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        valid = labels.ne(-100) if labels is not None else torch.ones_like(input_ids).bool()
        selected = logits[valid]
        loss = selected.square().mean() if selected.numel() else logits.sum() * 0
        return SimpleNamespace(
            loss=loss,
            logits=logits,
            hidden_states=(hidden,),
        )


def _batch():
    box, pose, seg, image = 51, 52, 53, 50
    input_ids = torch.tensor(
        [[1, box, pose, seg, image, image, image, image, 2, box, pose, seg, 3]]
    )
    labels = input_ids.clone()
    labels[:, :6] = -100
    mask = torch.zeros((1, 32, 32), dtype=torch.bool)
    mask[:, 8:24, 8:24] = True
    pose_target = torch.zeros((1, 17, 3))
    pose_target[..., 0] = 0.5
    pose_target[..., 1] = 0.5
    pose_target[..., 2] = 2
    return {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
        "labels": labels,
        "image_grid_thw": torch.tensor([[1, 4, 4]]),
        "box_targets": [torch.tensor([[0.2, 0.2, 0.8, 0.8]])],
        "pose_targets": [pose_target],
        "pose_areas": [torch.tensor([400.0])],
        "seg_targets": [mask],
        "box_person_indices": [torch.tensor([0])],
        "pose_person_indices": [torch.tensor([0])],
        "seg_person_indices": [torch.tensor([0])],
        "original_sizes": [(32, 32)],
    }


def test_full_multitask_forward_is_finite_and_backpropagates():
    base = FakeVLModel()
    model = Stage1V1Model(
        base_model=base,
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
    )
    assert model.pose_decoder_type == "direct_joint_query"
    assert model.pose_roi_enabled is False
    outputs = model(**_batch())
    assert torch.isfinite(outputs["loss"])
    for name in [
        "ce_loss",
        "box_l1",
        "box_giou",
        "box_quality",
        "pose_l1",
        "pose_oks",
        "pose_heatmap",
        "pose_visibility",
        "mask_bce",
        "mask_dice",
    ]:
        assert name in outputs
        assert torch.isfinite(outputs[name])
    assert outputs["valid_box_count"].item() == 1
    assert outputs["valid_pose_count"].item() == 1
    assert outputs["valid_seg_count"].item() == 1
    outputs["loss"].backward()
    assert base.embedding.weight.grad is not None
    assert torch.isfinite(base.embedding.weight.grad).all()


def test_weighted_causal_language_loss_downweights_unknown_tokens():
    logits = torch.tensor(
        [[[6.0, -6.0], [-6.0, 6.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    labels = torch.tensor([[-100, 0, 0]])
    full = weighted_causal_language_loss(
        logits, labels, torch.tensor([[0.0, 1.0, 1.0]])
    )
    downweighted = weighted_causal_language_loss(
        logits, labels, torch.tensor([[0.0, 1.0, 0.25]])
    )
    assert torch.isfinite(full)
    assert torch.isfinite(downweighted)
    assert downweighted < full


def test_forward_rejects_visual_grid_mismatch():
    model = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
    )
    batch = _batch()
    batch["image_grid_thw"] = torch.tensor([[1, 6, 6]])
    with pytest.raises(RuntimeError, match="visual token/grid mismatch"):
        model(**batch)


def test_forward_rejects_person_binding_out_of_range():
    model = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
    )
    batch = _batch()
    batch["pose_person_indices"] = [torch.tensor([2])]
    with pytest.raises(RuntimeError, match="pose person binding"):
        model(**batch)



def test_default_pose_loss_scale_is_vlm_stable():
    from human_lisa_qwen_v1.modeling_stage1_v1 import DEFAULT_LOSS_WEIGHTS

    assert DEFAULT_LOSS_WEIGHTS["pose_l1"] == 5.0
    assert DEFAULT_LOSS_WEIGHTS["pose_oks"] == 1.5
    assert DEFAULT_LOSS_WEIGHTS["pose_heatmap"] == 0.0


def test_pose_box_jitter_checkpoint_roundtrip_and_legacy_preservation(tmp_path):
    kwargs = {
        "hidden_size": 32,
        "token_ids": {"box": 51, "pose": 52, "seg": 53},
        "image_token_id": 50,
        "decoder_dim": 16,
        "decoder_heads": 4,
        "visual_merge_size": 2,
    }
    model = Stage1V1Model(
        base_model=FakeVLModel(),
        pose_box_jitter_enabled=True,
        pose_box_shift_factor=0.2,
        pose_box_shift_prob=0.4,
        pose_box_scale_range=(0.7, 1.3),
        pose_box_scale_prob=0.8,
        **kwargs,
    )
    model.save_task_heads(str(tmp_path / "jitter"))
    checkpoint = tmp_path / "jitter" / "stage1_task_heads.pt"

    restored = Stage1V1Model(base_model=FakeVLModel(), **kwargs)
    restored.load_task_heads(str(checkpoint))
    assert restored.pose_box_jitter_enabled is True
    assert restored.pose_box_shift_factor == pytest.approx(0.2)
    assert restored.pose_box_shift_prob == pytest.approx(0.4)
    assert restored.pose_box_scale_range == pytest.approx((0.7, 1.3))
    assert restored.pose_box_scale_prob == pytest.approx(0.8)

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state.pop("pose_box_jitter")
    legacy_path = tmp_path / "legacy_without_jitter.pt"
    torch.save(state, legacy_path)
    legacy_restored = Stage1V1Model(
        base_model=FakeVLModel(),
        pose_box_jitter_enabled=True,
        **kwargs,
    )
    legacy_restored.load_task_heads(str(legacy_path))
    assert legacy_restored.pose_box_jitter_enabled is True


def test_legacy_task_head_checkpoint_disables_pose_roi(tmp_path):
    model = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
        pose_decoder_type="coordinate",
    )
    model.save_task_heads(str(tmp_path / "new"))
    state = torch.load(
        tmp_path / "new" / "stage1_task_heads.pt",
        map_location="cpu",
        weights_only=False,
    )
    state.pop("pose_decoder_type")
    state.pop("pose_roi_enabled")
    state.pop("pose_joint_reference_enabled")
    state["pose_decoder"] = {
        name: value
        for name, value in state["pose_decoder"].items()
        if not name.startswith("box_project.")
        and name != "joint_reference_logits"
    }
    legacy_path = tmp_path / "legacy.pt"
    torch.save(state, legacy_path)

    restored = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
        pose_decoder_type="coordinate",
    )
    restored.load_task_heads(str(legacy_path))
    assert restored.pose_roi_enabled is False

    assert restored.pose_joint_reference_enabled is False

def test_task_gradient_scaling_preserves_forward_and_scales_backward():
    value = torch.tensor([2.0], requires_grad=True)
    scaled = scale_task_gradient(value, 0.1)
    torch.testing.assert_close(scaled, value)
    scaled.square().sum().backward()
    torch.testing.assert_close(value.grad, torch.tensor([0.4]))
    with pytest.raises(ValueError, match="task backbone gradient scale"):
        scale_task_gradient(value, -0.1)


def test_task_head_resume_can_preserve_runtime_training_config(tmp_path):
    source = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
        loss_weights={"ce": 1.0, "pose_l1": 5.0},
        task_backbone_grad_scale=0.1,
    )
    source.save_task_heads(str(tmp_path))
    runtime_weights = {
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
    resumed = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
        loss_weights=runtime_weights,
        task_backbone_grad_scale=0.0,
    )
    resumed.load_task_heads(
        str(tmp_path / "stage1_task_heads.pt"),
        restore_training_config=False,
    )
    assert resumed.loss_weights == runtime_weights
    assert resumed.task_backbone_grad_scale == 0.0




def test_task_head_resume_can_skip_pose_decoder(tmp_path):
    source = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
    )
    with torch.no_grad():
        for parameter in source.box_head.parameters():
            parameter.fill_(0.25)
        for parameter in source.pose_decoder.parameters():
            parameter.fill_(0.75)
        for parameter in source.mask_decoder.parameters():
            parameter.fill_(-0.25)
    source.save_task_heads(str(tmp_path))

    resumed = Stage1V1Model(
        base_model=FakeVLModel(),
        hidden_size=32,
        token_ids={"box": 51, "pose": 52, "seg": 53},
        image_token_id=50,
        decoder_dim=16,
        decoder_heads=4,
        visual_merge_size=2,
    )
    initial_pose = {
        name: value.detach().clone()
        for name, value in resumed.pose_decoder.state_dict().items()
    }
    resumed.load_task_heads(
        str(tmp_path / "stage1_task_heads.pt"),
        load_pose_decoder=False,
    )
    for name, value in resumed.pose_decoder.state_dict().items():
        torch.testing.assert_close(value, initial_pose[name])
    for value in resumed.box_head.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 0.25))
    for value in resumed.mask_decoder.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, -0.25))



def test_pose_reference_uses_detached_predicted_box_at_train_and_eval():
    predicted = torch.tensor([[0.1, 0.1, 0.4, 0.8]], requires_grad=True)
    target = torch.tensor([[0.2, 0.2, 0.6, 0.9]])
    train_reference = select_pose_reference_boxes(
        predicted,
        target,
        training=True,
    )
    eval_reference = select_pose_reference_boxes(
        predicted,
        target,
        training=False,
    )
    torch.testing.assert_close(train_reference, predicted.detach())
    torch.testing.assert_close(eval_reference, predicted.detach())
    assert not train_reference.requires_grad
    assert not eval_reference.requires_grad


