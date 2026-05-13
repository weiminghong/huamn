import pytest

pytest.importorskip("torch")

from human_slot_memory_pipeline.human_slot_memory.prompts import (
    build_stage3_prompt,
)
from human_slot_memory_pipeline.human_slot_memory.models.human_memory import build_human_memory_text
from human_slot_memory_pipeline.human_slot_memory.schema import HumanSample, PersonSlot, TaskMask


def test_stage3_prompt_mentions_box_pose_attr_only() -> None:
    prompt = build_stage3_prompt("Summarize the scene.")
    lower_prompt = prompt.lower()
    assert "box" in lower_prompt
    assert "pose" in lower_prompt
    assert "attribute" in lower_prompt
    assert "segmentation" not in lower_prompt


def test_build_human_memory_text_omits_segmentation_status() -> None:
    sample = HumanSample(
        sample_id="1",
        image_path="demo.jpg",
        width=640,
        height=480,
        slots=[
            PersonSlot(
                slot_id=0,
                box_xyxy=[64, 48, 192, 288],
                pose17_xyv=[[0.0, 0.0, 0.0]] * 17,
                attributes={"gender": 0, "age": 23},
                task_mask=TaskMask(box=True, pose=True, attr=True),
            )
        ],
    )
    text = build_human_memory_text(sample)
    assert "<segmentation>" not in text
    assert "<pose_visible>0</pose_visible>" in text
    assert "<gender>0</gender>" in text
    assert "<age>23</age>" in text
