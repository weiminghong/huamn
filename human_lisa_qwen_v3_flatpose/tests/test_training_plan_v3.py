from human_lisa_qwen_v3_flatpose.training_v3 import (
    accumulation_group_state,
    optimizer_steps_for_micro_plan,
)


def test_full_plan_closes_odd_epoch_boundary():
    assert optimizer_steps_for_micro_plan(8015, 8015, 2) == 4008
    assert optimizer_steps_for_micro_plan(8015, 8015 * 3, 2) == 12024
    divisor, boundary = accumulation_group_state(
        batches_in_epoch=8015, micro_step=8015, micro_steps_per_epoch=8015,
        target_micro_steps=8015 * 3, grad_accum_steps=2,
    )
    assert divisor == 1
    assert boundary is True


def test_new_epoch_starts_a_fresh_accumulation_group():
    divisor, boundary = accumulation_group_state(
        batches_in_epoch=1, micro_step=8016, micro_steps_per_epoch=8015,
        target_micro_steps=8015 * 3, grad_accum_steps=2,
    )
    assert divisor == 2
    assert boundary is False


def test_regular_pair_uses_full_divisor_and_steps():
    divisor, boundary = accumulation_group_state(
        batches_in_epoch=2, micro_step=2, micro_steps_per_epoch=8015,
        target_micro_steps=8015 * 3, grad_accum_steps=2,
    )
    assert divisor == 2
    assert boundary is True
