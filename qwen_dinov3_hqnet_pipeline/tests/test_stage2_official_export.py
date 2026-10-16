
import numpy as np
import torch

from qwen_dinov3_hqnet_pipeline.tools.eval_stage2_qwen_dino_hqnet_native import (
    stage2_result_to_official_predictions,
)


def test_stage2_result_to_official_predictions_exports_bbox_keypoints_attr_and_mask():
    mask = np.zeros((12, 14), dtype=bool)
    mask[2:9, 3:11] = True
    det = np.zeros((1, 4 + 1 + 1 + 1 + 34), dtype=np.float32)
    det[0, :7] = [10, 20, 30, 50, 0.91, 0.25, 34.0]
    for j in range(17):
        det[0, 7 + 2 * j] = 10 + j
        det[0, 7 + 2 * j + 1] = 20 + j
    result = {
        "det": [[det]],
        "ins_results": [([det[:, :5]], [[mask]])],
    }

    records = stage2_result_to_official_predictions(result, image_id=123, score_thr=0.3)

    assert len(records) == 1
    rec = records[0]
    assert rec["image_id"] == 123
    assert rec["category_id"] == 1
    assert rec["bbox"] == [10.0, 20.0, 20.0, 30.0]
    assert abs(rec["score"] - 0.91) < 1e-5
    assert rec["gender"] == 0
    assert rec["age"] == 34.0
    assert len(rec["keypoints"]) == 51
    assert rec["keypoints"][:6] == [10.0, 20.0, 1.0, 11.0, 21.0, 1.0]
    assert "segmentation" in rec
    assert rec["segmentation"]["size"] == [12, 14]
    assert isinstance(rec["segmentation"]["counts"], str)



def test_stage2_result_to_official_predictions_exports_head_seg_structure():
    mask = torch.zeros((1, 8, 9), dtype=torch.bool)
    mask[0, 1:5, 2:7] = True
    det = torch.zeros((1, 4 + 1 + 1 + 1 + 34), dtype=torch.float32)
    det[0, :7] = torch.tensor([1, 2, 6, 8, 0.77, 0.8, 18.0])
    for j in range(17):
        det[0, 7 + 2 * j] = 1 + j
        det[0, 7 + 2 * j + 1] = 2 + j
    result = {
        "det": det,
        "seg": [{"ins_results": (torch.tensor([0]), torch.tensor([[1, 2, 6, 8, 0.66]], dtype=torch.float32), mask)}],
    }

    records = stage2_result_to_official_predictions(result, image_id=7, score_thr=0.3)

    assert len(records) == 1
    assert records[0]["bbox"] == [1.0, 2.0, 5.0, 6.0]
    assert abs(records[0]["score"] - 0.77) < 1e-5
    assert records[0]["segmentation"]["size"] == [8, 9]


if __name__ == "__main__":
    test_stage2_result_to_official_predictions_exports_bbox_keypoints_attr_and_mask()
    test_stage2_result_to_official_predictions_exports_head_seg_structure()
    print("ok")
