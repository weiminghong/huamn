import pytest

torch = pytest.importorskip("torch")

from human_slot_memory_pipeline.human_slot_memory.losses.partial_label import PartialLabelLoss
from human_slot_memory_pipeline.human_slot_memory.models.dense_student import DenseHumanStudent
from human_slot_memory_pipeline.human_slot_memory.schema import HumanBatch, HumanSample, PersonSlot, TaskMask


def test_dense_student_outputs_exclude_segmentation() -> None:
    model = DenseHumanStudent(dinov3_model_path="unused", dry_run=True)
    outputs = model(torch.randn(2, 3, 224, 224))

    assert "box_xyxy_norm" in outputs
    assert "pose17_xyv_norm" in outputs
    assert "gender_logits" in outputs
    assert "age" in outputs
    assert "mask_logits" not in outputs


def test_partial_label_loss_excludes_segmentation_losses() -> None:
    model = DenseHumanStudent(dinov3_model_path="unused", dry_run=True)
    outputs = model(torch.randn(1, 3, 224, 224))
    batch = HumanBatch(
        samples=[
            HumanSample(
                sample_id="0",
                image_path="demo.jpg",
                width=640,
                height=480,
                slots=[
                    PersonSlot(
                        slot_id=0,
                        box_xyxy=[64.0, 48.0, 192.0, 288.0],
                        pose17_xyv=[[0.0, 0.0, 0.0] for _ in range(17)],
                        attributes={"gender": 0, "age": 23},
                        task_mask=TaskMask(box=True, pose=False, attr=True),
                    )
                ],
            )
        ]
    )

    losses = PartialLabelLoss()(outputs, batch)

    assert "loss_box" in losses
    assert "loss_pose" in losses
    assert "loss_validity" in losses
    assert "loss_seg_bce" not in losses
    assert "loss_seg_dice" not in losses
