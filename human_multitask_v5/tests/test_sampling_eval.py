import pytest

from human_multitask_v5.eval_protocols import (
    EvaluationPlanError,
    assert_no_cross_protocol_average,
    build_plan,
)
from human_multitask_v5.sampling import (
    BalancedPacketSampler,
    build_packet_index,
)
from human_multitask_v5.tests.test_schema_protocol import record


def test_balanced_sampler_is_deterministic_and_eligible():
    first = record()
    second = record()
    second["uid"] = "mhp:train:2"
    second["dataset"] = "mhp_v2"
    second["image"]["coco_id"] = None
    second["persons"][0]["labels"]["age"] = {"valid": False}
    index = build_packet_index([first, second])
    weights = {"box": 1, "pose": 1, "segment": 1, "all": 0.5}
    sampler = BalancedPacketSampler(
        index, weights, epoch_size=100, seed=17
    )
    packets_a, audit_a = sampler.epoch(2)
    packets_b, audit_b = sampler.epoch(2)
    assert packets_a == packets_b
    assert audit_a == audit_b
    assert sum(audit_a.task_counts.values()) == 100
    assert all(packet.task in weights for packet in packets_a)


def test_eval_protocols_remain_separate():
    plan = build_plan(
        {
            "coco_unihuman": ["box", "pose"],
            "mhp_v2": ["parsing"],
        }
    )
    assert len(plan) == 3
    with pytest.raises(EvaluationPlanError):
        assert_no_cross_protocol_average(
            [
                {"metric": "AP", "comparable_group": "coco_v5"},
                {"metric": "AP", "comparable_group": "other"},
            ]
        )
