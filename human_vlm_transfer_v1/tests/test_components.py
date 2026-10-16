import torch

from human_vlm_transfer_v1.modeling import (
    BoxConditionedPoseHead,
    HumanQueryDecoder,
    SpatialAdapter,
    crop_feature_rois,
    generalized_box_iou,
    unshuffle_premerger_tokens,
)


def test_premerger_unshuffle_restores_raster_order():
    height, width, merge, channels = 4, 6, 2, 3
    raster = torch.arange(height * width * channels).reshape(
        height, width, channels
    )
    grouped = (
        raster.reshape(height // merge, merge, width // merge, merge, channels)
        .permute(0, 2, 1, 3, 4)
        .reshape(height * width, channels)
    )
    restored = unshuffle_premerger_tokens(
        grouped,
        grid_thw=(1, height, width),
        merge_size=merge,
    )
    assert torch.equal(restored, raster.permute(2, 0, 1))


def test_generalized_iou_identity_is_one():
    boxes = torch.tensor([[0.1, 0.2, 0.7, 0.9], [0.0, 0.0, 0.2, 0.3]])
    values = generalized_box_iou(boxes, boxes)
    assert torch.allclose(values.diag(), torch.ones(2), atol=1e-6)


def test_spatial_adapter_and_query_decoder_backpropagate():
    adapter = SpatialAdapter(32, 64, 3)
    decoder = HumanQueryDecoder(64, 10, 2, 8)
    maps = [torch.randn(32, 8, 10, requires_grad=True) for _ in range(3)]
    p16, p8, p4 = adapter(maps)
    queries = decoder(p16, torch.randn(1, 64))
    loss = queries.square().mean() + p8.square().mean() + p4.square().mean()
    loss.backward()
    assert queries.shape == (1, 10, 64)
    assert p8.shape[-2:] == (16, 20)
    assert p4.shape[-2:] == (32, 40)
    assert all(value.grad is not None for value in maps)


def test_roi_pose_features_backpropagate():
    feature = torch.randn(1, 64, 32, 40, requires_grad=True)
    boxes = torch.tensor(
        [[0.1, 0.2, 0.7, 0.9], [0.3, 0.1, 0.8, 0.6]],
        dtype=torch.float32,
    )
    rois = crop_feature_rois(feature, boxes, output_size=(16, 12))
    pose_head = BoxConditionedPoseHead(64, num_heads=8, num_layers=1)
    relative, visibility, heatmaps = pose_head(torch.randn(2, 64), rois)
    loss = (
        relative.square().mean()
        + visibility.square().mean()
        + heatmaps.square().mean()
    )
    loss.backward()
    assert rois.shape == (2, 64, 16, 12)
    assert relative.shape == (2, 17, 2)
    assert heatmaps.shape == (2, 17, 16, 12)
    assert feature.grad is not None
