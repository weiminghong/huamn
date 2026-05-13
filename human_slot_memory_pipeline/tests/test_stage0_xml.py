from human_slot_memory_pipeline.human_slot_memory.stage0_xml import (
    answer_block_count,
    contains_think_tokens,
    prediction_needs_repair,
    truncate_to_first_answer,
)


def _pose17() -> str:
    return ", ".join(["[0,0,0]"] * 17)


def test_truncate_to_first_answer_discards_duplicate_and_think() -> None:
    raw = (
        "<answer><person><box>[1,2,3,4]</box><pose>"
        + _pose17()
        + "</pose><gender>0</gender><age>20</age><score>0.9</score></person></answer>"
        + "</think><answer><person></person></answer>"
    )
    cleaned = truncate_to_first_answer(raw)
    assert cleaned.count("<answer>") == 1
    assert cleaned.endswith("</answer>")
    assert "</think>" not in cleaned


def test_prediction_needs_repair_for_think_or_duplicate_answer() -> None:
    raw = (
        "<answer><person><box>[1,2,3,4]</box><pose>"
        + _pose17()
        + "</pose><gender>0</gender><age>20</age><score>0.9</score></person></answer>"
        + "<think>hidden</think>"
    )
    need_repair, cleaned = prediction_needs_repair(raw, image_size=(1000, 1000))
    assert contains_think_tokens(raw) is True
    assert answer_block_count(cleaned) == 1
    assert need_repair is True


def test_prediction_needs_repair_for_invalid_pose_length() -> None:
    raw = (
        "<answer><person><box>[1,2,3,4]</box>"
        "<pose>[1,2,3,4], [5,6,7,8]</pose>"
        "<gender>0</gender><age>20</age><score>0.9</score></person></answer>"
    )
    need_repair, _ = prediction_needs_repair(raw, image_size=(1000, 1000))
    assert need_repair is True


def test_prediction_needs_no_repair_for_valid_single_answer() -> None:
    raw = (
        "<answer><person><box>[10,20,300,400]</box><pose>"
        + _pose17()
        + "</pose><gender>1</gender><age>25</age><score>0.98</score></person></answer>"
    )
    need_repair, cleaned = prediction_needs_repair(raw, image_size=(1000, 1000))
    assert contains_think_tokens(cleaned) is False
    assert answer_block_count(cleaned) == 1
    assert need_repair is False
