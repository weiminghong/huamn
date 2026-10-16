import json

import numpy as np
import torch

from human_lisa_qwen_v4.losses_v4 import task_normalized_causal_loss
from human_lisa_qwen_v4.matching_v4 import hungarian_box_match
from human_lisa_qwen_v4.training_v4 import (
    EpochRandomSampler,
    clip_gradients_by_group,
    scheduled_route_loss_weights,
    set_step_learning_rates,
    suppress_zero_lr_gradients,
)
from human_lisa_qwen_v4.schema_v4 import (
    apply_chat_template_v4,
    canonicalize_person_ids_v4,
    validate_prediction_object_v4,
)
from human_lisa_qwen_v4.targets_v4 import (
    TEXT_TASK_TO_ID,
    assign_token_tasks,
    serialize_target_v4,
)


class _ThinkingProcessor:
    def __init__(self, reject_flag=False):
        self.reject_flag = reject_flag
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append(dict(kwargs))
        if self.reject_flag and "enable_thinking" in kwargs:
            raise TypeError("legacy template")
        return "closed-think-prefix" if kwargs.get("enable_thinking") is False else "fallback-prefix"


def test_chat_template_disables_thinking_with_legacy_fallback():
    processor = _ThinkingProcessor()
    assert apply_chat_template_v4(processor, []) == "closed-think-prefix"
    assert processor.calls == [
        {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
    ]
    legacy = _ThinkingProcessor(reject_flag=True)
    assert apply_chat_template_v4(legacy, []) == "fallback-prefix"
    assert "enable_thinking" not in legacy.calls[-1]


def _instance(person_id, box, *, pose=True, seg=True, gender=True, age=True):
    mask = np.zeros((20, 40), dtype=bool)
    mask[2:10, 3:12] = True
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = 8
    keypoints[:, 1] = 7
    keypoints[:, 2] = 2
    return {
        "person_id": person_id,
        "source_ann_id": 100 + person_id,
        "bbox_xyxy": box,
        "area": 200.0,
        "pose_valid": pose,
        "keypoints": keypoints,
        "seg_valid": seg,
        "mask": mask,
        "gender_valid": gender,
        "gender": 0,
        "age_valid": age,
        "age": 30.0,
    }


def test_serializer_binds_dense_targets_without_fixed_slots():
    target = serialize_target_v4(
        [
            _instance(1, [0, 0, 20, 20], pose=True, seg=False),
            _instance(
                2,
                [20, 0, 40, 20],
                pose=False,
                seg=True,
                gender=False,
                age=False,
            ),
        ],
        width=40,
        height=20,
    )
    value = json.loads(target.text)
    validate_prediction_object_v4(value)
    assert len(value["persons"]) == 2
    assert value["persons"][0]["box"] == [0, 0, 500, 1000]
    assert target.pose_person_indices == [0]
    assert target.seg_person_indices == [1]
    assert target.person_count == 2
    missing = {
        (span.task, span.person_index)
        for span in target.spans
        if not span.valid
    }
    assert missing == {("gender", 1), ("age", 1)}


def test_task_assignment_prefers_semantic_span():
    target = serialize_target_v4(
        [_instance(1, [2, 3, 20, 18])], width=40, height=20
    )
    offsets = [(span.start, span.end) for span in target.spans]
    task_ids, valid = assign_token_tasks(offsets, target.spans)
    expected = [TEXT_TASK_TO_ID[span.task] for span in target.spans]
    assert task_ids == expected
    assert all(valid)


def test_task_normalization_does_not_reward_longer_task():
    logits = torch.zeros(1, 5, 3)
    labels = torch.tensor([[-100, 1, 1, 1, 1]])
    task_ids = torch.tensor(
        [
            [
                0,
                TEXT_TASK_TO_ID["box"],
                TEXT_TASK_TO_ID["box"],
                TEXT_TASK_TO_ID["box"],
                TEXT_TASK_TO_ID["gender"],
            ]
        ]
    )
    loss, components, counts = task_normalized_causal_loss(
        logits, labels, task_ids
    )
    assert torch.isfinite(loss)
    assert counts["box"] == 3
    assert counts["gender"] == 1
    assert torch.allclose(components["box"], components["gender"])


def test_masked_value_has_no_loss():
    logits = torch.randn(1, 4, 7, requires_grad=True)
    labels = torch.tensor([[-100, 2, -100, 4]])
    task_ids = torch.tensor(
        [
            [
                0,
                TEXT_TASK_TO_ID["box"],
                TEXT_TASK_TO_ID["gender"],
                TEXT_TASK_TO_ID["structure"],
            ]
        ]
    )
    loss, _, counts = task_normalized_causal_loss(logits, labels, task_ids)
    loss.backward()
    assert counts["gender"] == 0
    assert torch.isfinite(loss)


def test_route_loss_weight_warmup_is_bounded_and_reaches_final():
    kwargs = {
        "start_weight": 0.02,
        "final_pose_weight": 0.25,
        "final_seg_weight": 0.15,
        "hold_steps": 100,
        "warmup_steps": 400,
    }
    start = scheduled_route_loss_weights(optimizer_step=0, **kwargs)
    held = scheduled_route_loss_weights(optimizer_step=99, **kwargs)
    first_ramp = scheduled_route_loss_weights(optimizer_step=100, **kwargs)
    middle = scheduled_route_loss_weights(optimizer_step=299, **kwargs)
    final = scheduled_route_loss_weights(optimizer_step=499, **kwargs)
    after = scheduled_route_loss_weights(optimizer_step=900, **kwargs)
    assert start == held == {"pose_route": 0.02, "seg_route": 0.02}
    assert first_ramp["pose_route"] > held["pose_route"]
    assert held["pose_route"] < middle["pose_route"] < final["pose_route"]
    assert np.isclose(final["pose_route"], 0.25)
    assert np.isclose(final["seg_route"], 0.15)
    assert final == after


def test_visual_learning_rate_thaw_is_continuous():
    visual = torch.nn.Parameter(torch.ones(1))
    head = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.SGD(
        [
            {
                "params": [visual],
                "lr": 1e-5,
                "initial_lr": 1e-5,
                "family": "visual_lora",
                "group_name": "visual",
            },
            {
                "params": [head],
                "lr": 2e-4,
                "initial_lr": 2e-4,
                "family": "pose_decoder",
                "group_name": "head",
            },
        ]
    )
    kwargs = {
        "total_steps": 8016,
        "warmup_steps": 240,
        "decoder_warmup_steps": 200,
        "visual_warmup_steps": 200,
    }
    before = set_step_learning_rates(
        optimizer, optimizer_step=199, **kwargs
    )
    first = set_step_learning_rates(
        optimizer, optimizer_step=200, **kwargs
    )
    middle = set_step_learning_rates(
        optimizer, optimizer_step=299, **kwargs
    )
    full = set_step_learning_rates(
        optimizer, optimizer_step=399, **kwargs
    )
    assert before["visual"] == 0.0
    assert 0.0 < first["visual"] < middle["visual"] < full["visual"]
    assert first["visual"] < 1e-7
    assert head.grad is None


def test_zero_lr_groups_do_not_create_optimizer_state_or_clip_active_grads():
    frozen = torch.nn.Parameter(torch.ones(2))
    active = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [frozen],
                "lr": 0.0,
                "initial_lr": 1e-5,
                "family": "merger",
                "group_name": "merger",
            },
            {
                "params": [active],
                "lr": 1e-4,
                "initial_lr": 1e-4,
                "family": "pose_decoder",
                "group_name": "head",
            },
        ]
    )
    frozen.grad = torch.tensor([30.0, 40.0])
    active.grad = torch.tensor([0.3, 0.4])
    stats = suppress_zero_lr_gradients(optimizer)
    assert frozen.grad is None
    assert torch.allclose(active.grad, torch.tensor([0.3, 0.4]))
    assert stats["families"]["merger"]["parameters"] == 1
    optimizer.step()
    assert frozen not in optimizer.state
    assert active in optimizer.state


def test_person_id_repairs_are_auditable_and_schema_valid():
    value = {
        "persons": [
            {
                "person_id": 7,
                "box": [0, 0, 100, 100],
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 0,
                "age": 20,
            },
            {
                "person_id": 7,
                "box": [200, 0, 300, 100],
                "pose": "<pose>",
                "seg": "<seg>",
                "gender": 1,
                "age": 30,
            },
        ]
    }
    assert canonicalize_person_ids_v4(value) == 2
    assert [person["person_id"] for person in value["persons"]] == [1, 2]
    validate_prediction_object_v4(value)


def test_token_gradient_clipping_does_not_shrink_dense_gradients():
    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.token_rows = torch.nn.Linear(2, 1, bias=False)
            self.pose_decoder = torch.nn.Linear(2, 1, bias=False)

    model = ToyModel()
    model.token_rows.weight.grad = torch.tensor([[30.0, 40.0]])
    model.pose_decoder.weight.grad = torch.tensor([[0.3, 0.4]])
    stats = clip_gradients_by_group(
        model,
        token_rows_max_norm=1.0,
        non_token_max_norm=1.0,
    )
    assert np.isclose(stats["pre_clip_norms"]["token_rows"], 50.0)
    assert np.isclose(stats["pre_clip_norms"]["non_token"], 0.5)
    assert np.isclose(model.token_rows.weight.grad.norm().item(), 1.0)
    assert np.isclose(model.pose_decoder.weight.grad.norm().item(), 0.5)
    assert np.isclose(stats["clip_scales"]["token_rows"], 0.02)
    assert np.isclose(stats["clip_scales"]["non_token"], 1.0)


def test_true_hungarian_matching_is_one_to_one():
    matches = hungarian_box_match(
        [[0, 0, 10, 10], [20, 0, 30, 10], [0, 0, 9, 9]],
        [[0, 0, 10, 10], [20, 0, 30, 10]],
        minimum_iou=0.5,
    )
    assert len(matches) == 2
    assert len({match.prediction_index for match in matches}) == 2
    assert len({match.target_index for match in matches}) == 2


def test_epoch_random_sampler_resume_matches_full_suffix():
    data = list(range(17))
    full = EpochRandomSampler(data, seed=23)
    full.set_epoch(4)
    full_order = list(full)

    resumed = EpochRandomSampler(data, seed=23)
    resumed.set_epoch(4, start_index=9)
    assert list(resumed) == full_order[9:]
    assert len(resumed) == len(data) - 9
