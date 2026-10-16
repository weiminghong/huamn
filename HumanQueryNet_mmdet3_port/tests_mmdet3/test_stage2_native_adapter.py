from __future__ import annotations

from compat.stage2_humanv2_adapter import build_humanv2_stage2_batch

ANN = "/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
IMG_ROOT = "/share2/program/weiminghong/coco/val2017"


def main() -> None:
    batch = build_humanv2_stage2_batch(
        ann_file=ANN,
        image_root=IMG_ROOT,
        image_size=128,
        max_instances=4,
        require_keypoints=True,
        device="cpu",
    )
    assert batch["img"].shape == (1, 3, 128, 128)
    assert len(batch["img_metas"]) == 1
    meta = batch["img_metas"][0]
    assert meta["img_shape"] == (128, 128, 3)
    assert meta["pad_shape"] == (128, 128, 3)
    assert meta["ori_shape"][0] > 0 and meta["ori_shape"][1] > 0

    assert len(batch["gt_bboxes"]) == 1
    n = batch["gt_bboxes"][0].shape[0]
    assert 0 < n <= 4
    assert batch["gt_bboxes"][0].shape == (n, 4)
    assert batch["gt_labels"][0].shape == (n,)
    assert batch["gt_attributes"][0].shape == (n, 2)
    assert batch["gt_keypoints"][0].shape == (n, 51)
    assert batch["gt_areas"][0].shape == (n,)
    assert batch["gt_valids"][0].shape == (n, 8)
    assert batch["gt_smpl_pose"][0].shape == (n, 72)
    assert batch["gt_smpl_betas"][0].shape == (n, 10)
    assert hasattr(batch["gt_masks"][0], "pad")
    assert hasattr(batch["gt_masks"][0], "to_tensor")

    boxes = batch["gt_bboxes"][0]
    assert bool((boxes[:, 2:] > boxes[:, :2]).all())
    assert bool((batch["gt_valids"][0][:, 1] == 1).all())
    assert int(batch["gt_valids"][0][:, 2].sum().item()) >= 1
    print("stage2_native_adapter_ok", batch["sample_info"]["img_id"], n)


if __name__ == "__main__":
    main()
