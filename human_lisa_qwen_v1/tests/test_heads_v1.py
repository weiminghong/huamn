import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.heads_v1 import (
    BoxQualityHead,
    DirectJointQueryPoseDecoder,
    MultiPersonPoseDecoder,
    PersonMaskDecoder,
    TopDownHeatmapPoseDecoder,
    generalized_box_iou_loss,
    jitter_pose_boxes,
    keypoints_inside_boxes,
    keypoint_heatmap_mse_loss,
    mask_bce_dice_loss,
    oks_loss,
    roi_relative_pose_l1,
    require_exact_token_states,
    roi_pool_visual_tokens,
    sample_reference_features,
)


def test_box_head_outputs_valid_boxes_quality_and_gradients():
    torch.manual_seed(3)
    head = BoxQualityHead(hidden_size=32, decoder_dim=16)
    states = torch.randn(5, 32, requires_grad=True)
    boxes, quality_logits = head(states)
    assert boxes.shape == (5, 4)
    assert quality_logits.shape == (5,)
    assert torch.all(boxes >= 0) and torch.all(boxes <= 1)
    assert torch.all(boxes[:, 2] > boxes[:, 0])
    assert torch.all(boxes[:, 3] > boxes[:, 1])
    assert torch.isfinite(quality_logits).all()
    (boxes.sum() + quality_logits.sum()).backward()
    assert states.grad is not None
    assert torch.isfinite(states.grad).all()


def test_generalized_iou_loss_is_zero_for_identical_boxes():
    boxes = torch.tensor(
        [[0.1, 0.2, 0.7, 0.8], [0.0, 0.0, 0.2, 0.3]], dtype=torch.float32
    )
    assert generalized_box_iou_loss(boxes, boxes).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_direct_joint_query_pose_decoder_uses_full_image_features():
    torch.manual_seed(23)
    decoder = DirectJointQueryPoseDecoder(
        hidden_size=32,
        visual_hidden_size=32,
        decoder_dim=32,
        num_heads=4,
        num_layers=2,
    )
    pose_states = torch.randn(2, 32, requires_grad=True)
    parent_states = torch.randn(2, 32, requires_grad=True)
    visual_tokens = torch.randn(1, 16, 32, requires_grad=True)
    boxes = torch.tensor(
        [[0.05, 0.10, 0.45, 0.90], [0.55, 0.10, 0.95, 0.90]],
        dtype=torch.float32,
    )
    xy, visibility = decoder(
        pose_states,
        parent_states,
        visual_tokens,
        spatial_shape=(4, 4),
        person_boxes=boxes,
    )
    assert xy.shape == (2, 17, 2)
    assert visibility.shape == (2, 17, 3)
    assert torch.all(xy >= 0.0) and torch.all(xy <= 1.0)
    assert not torch.allclose(xy[0], xy[1])
    (xy.sum() + visibility.sum()).backward()
    assert visual_tokens.grad is not None
    assert torch.isfinite(visual_tokens.grad).all()

def test_reference_feature_sampler_reads_joint_local_features():
    feature_map = torch.arange(9, dtype=torch.float32).reshape(1, 1, 3, 3)
    references = torch.tensor(
        [[[0.5, 0.5], [1.0 / 6.0, 1.0 / 6.0]]],
        dtype=torch.float32,
    )
    offsets = torch.tensor(
        [[
            [[0.0, 0.0], [1.0 / 3.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0 / 3.0]],
        ]],
        dtype=torch.float32,
    )
    weights = torch.tensor(
        [[[0.25, 0.75], [0.5, 0.5]]],
        dtype=torch.float32,
    )
    sampled = sample_reference_features(
        feature_map,
        references,
        offsets,
        weights,
    )
    assert sampled.shape == (1, 2, 1)
    torch.testing.assert_close(
        sampled[..., 0],
        torch.tensor([[4.75, 1.5]]),
        atol=1e-6,
        rtol=0.0,
    )


def test_direct_decoder_uses_reference_point_refinement_layers():
    decoder = DirectJointQueryPoseDecoder(
        hidden_size=32,
        visual_hidden_size=32,
        decoder_dim=32,
        num_heads=4,
        num_layers=2,
    )
    assert len(decoder.refinement_layers) == 2
    assert all(
        hasattr(layer, "sampling_offsets")
        for layer in decoder.refinement_layers
    )



def test_pose_decoder_supports_zero_one_and_multiple_people():
    torch.manual_seed(4)
    decoder = MultiPersonPoseDecoder(
        hidden_size=32, visual_hidden_size=32, decoder_dim=16, num_heads=4
    )
    visual = torch.randn(1, 12, 32)
    for persons in [0, 1, 4]:
        token_states = torch.randn(persons, 32, requires_grad=True)
        parent_states = torch.randn(persons, 32, requires_grad=True)
        xy, visibility = decoder(token_states, parent_states, visual)
        assert xy.shape == (persons, 17, 2)
        assert visibility.shape == (persons, 17, 3)
        if persons:
            (xy.sum() + visibility.sum()).backward()
            assert token_states.grad is not None
            assert parent_states.grad is not None


def test_oks_loss_perfect_pose_and_invisible_keypoints():
    prediction = torch.tensor(
        [[[0.2, 0.3], [0.8, 0.7]]], dtype=torch.float32
    )
    target = prediction.clone()
    visibility = torch.tensor([[2, 0]], dtype=torch.long)
    area = torch.tensor([1000.0])
    image_sizes = torch.tensor([[100.0, 200.0]])
    perfect = oks_loss(
        prediction,
        target,
        visibility,
        area,
        image_sizes,
        sigmas=torch.tensor([0.026, 0.025]),
    )
    assert perfect.item() == pytest.approx(0.0, abs=1e-6)
    target[:, 1] = 0.0
    still_perfect = oks_loss(
        prediction,
        target,
        visibility,
        area,
        image_sizes,
        sigmas=torch.tensor([0.026, 0.025]),
    )
    assert still_perfect.item() == pytest.approx(0.0, abs=1e-6)


def test_mask_decoder_and_loss_support_multiple_people():
    torch.manual_seed(5)
    decoder = PersonMaskDecoder(
        hidden_size=32, visual_hidden_size=32, decoder_dim=16
    )
    seg_states = torch.randn(3, 32, requires_grad=True)
    parent_states = torch.randn(3, 32, requires_grad=True)
    visual_tokens = torch.randn(1, 12, 32)
    logits = decoder(
        seg_states, parent_states, visual_tokens, spatial_shape=(3, 4)
    )
    assert logits.shape[0] == 3
    assert logits.shape[-2:] == (12, 16)
    targets = torch.zeros((3, 24, 32), dtype=torch.bool)
    targets[:, 4:20, 8:24] = True
    bce, dice = mask_bce_dice_loss(logits, targets)
    assert torch.isfinite(bce) and torch.isfinite(dice)
    (bce + dice).backward()
    assert seg_states.grad is not None
    assert parent_states.grad is not None


def test_perfect_mask_logits_have_near_zero_losses():
    target = torch.tensor([[[0, 1], [1, 0]]], dtype=torch.bool)
    logits = torch.where(target, torch.tensor(20.0), torch.tensor(-20.0))
    bce, dice = mask_bce_dice_loss(logits, target)
    assert bce.item() < 1e-6
    assert dice.item() < 1e-6


def test_exact_token_binding_raises_instead_of_dropping_targets():
    input_ids = torch.tensor([[1, 7, 3, 7, 4]])
    hidden = torch.randn(1, 5, 8)
    labels = torch.tensor([[-100, 7, 3, 7, 4]])
    states = require_exact_token_states(
        input_ids=input_ids,
        hidden=hidden,
        labels=labels,
        token_id=7,
        expected_counts=[2],
        task_name="box",
    )
    assert states[0].shape == (2, 8)
    with pytest.raises(RuntimeError, match="box token/target mismatch"):
        require_exact_token_states(
            input_ids=input_ids,
            hidden=hidden,
            labels=labels,
            token_id=7,
            expected_counts=[3],
            task_name="box",
        )



def test_oks_loss_is_linear_and_bounded_for_bad_pose():
    prediction = torch.zeros((1, 17, 2), dtype=torch.float32)
    target = torch.ones((1, 17, 2), dtype=torch.float32)
    visibility = torch.full((1, 17), 2, dtype=torch.long)
    loss = oks_loss(
        prediction,
        target,
        visibility,
        torch.tensor([100.0]),
        torch.tensor([[100.0, 100.0]]),
    )
    assert 0.0 <= loss.item() <= 1.0
    assert loss.item() > 0.99


def test_roi_pool_visual_tokens_is_person_specific():
    rows = torch.arange(4, dtype=torch.float32)[:, None].expand(4, 4)
    cols = torch.arange(4, dtype=torch.float32)[None, :].expand(4, 4)
    features = torch.stack([rows, cols], dim=-1).reshape(1, 16, 2)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.5, 1.0], [0.5, 0.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    pooled = roi_pool_visual_tokens(
        features,
        spatial_shape=(4, 4),
        boxes=boxes,
        output_size=3,
    )
    assert pooled.shape == (2, 9, 2)
    assert pooled[0, :, 1].mean() < pooled[1, :, 1].mean()
    empty = roi_pool_visual_tokens(
        features,
        spatial_shape=(4, 4),
        boxes=boxes[:0],
        output_size=3,
    )
    assert empty.shape == (0, 9, 2)


def test_pose_box_jitter_identity_and_valid_random_boxes():
    boxes = torch.tensor(
        [[0.10, 0.20, 0.50, 0.80], [0.00, 0.00, 1.00, 1.00]],
        dtype=torch.float32,
    )
    identity = jitter_pose_boxes(
        boxes,
        shift_factor=0.0,
        shift_prob=1.0,
        scale_range=(1.0, 1.0),
        scale_prob=1.0,
    )
    assert torch.equal(identity, boxes)

    torch.manual_seed(17)
    jittered = jitter_pose_boxes(
        boxes,
        shift_factor=0.16,
        shift_prob=1.0,
        scale_range=(0.5, 1.5),
        scale_prob=1.0,
    )
    assert jittered.shape == boxes.shape
    assert torch.all(jittered >= 0.0)
    assert torch.all(jittered <= 1.0)
    assert torch.all(jittered[:, 2:] > jittered[:, :2])
    assert not torch.allclose(jittered, boxes)


def test_keypoints_inside_boxes_marks_jittered_roi_exclusions():
    points = torch.tensor(
        [[[0.20, 0.20], [0.50, 0.50], [0.80, 0.80]]],
        dtype=torch.float32,
    )
    boxes = torch.tensor([[0.25, 0.25, 0.75, 0.75]], dtype=torch.float32)
    inside = keypoints_inside_boxes(points, boxes)
    assert inside.tolist() == [[False, True, False]]


def test_pose_decoder_accepts_person_roi_tokens_and_boxes():
    torch.manual_seed(8)
    decoder = MultiPersonPoseDecoder(
        hidden_size=32, visual_hidden_size=32, decoder_dim=16, num_heads=4
    )
    pose_states = torch.randn(2, 32)
    parent_states = torch.randn(2, 32)
    roi_tokens = torch.randn(2, 9, 32)
    boxes = torch.tensor(
        [[0.0, 0.0, 0.4, 1.0], [0.6, 0.0, 1.0, 1.0]]
    )
    xy, visibility = decoder(
        pose_states,
        parent_states,
        roi_tokens,
        person_boxes=boxes,
    )
    assert xy.shape == (2, 17, 2)
    assert visibility.shape == (2, 17, 3)
    assert not torch.allclose(xy[0], xy[1])
    assert torch.all(xy[0, :, 0] >= boxes[0, 0])
    assert torch.all(xy[0, :, 0] <= boxes[0, 2])
    assert torch.all(xy[0, :, 1] >= boxes[0, 1])
    assert torch.all(xy[0, :, 1] <= boxes[0, 3])
    assert torch.all(xy[1, :, 0] >= boxes[1, 0])


def test_roi_relative_pose_l1_weights_equal_local_errors_equally():
    boxes = torch.tensor(
        [[0.0, 0.0, 0.2, 0.2], [0.0, 0.0, 0.8, 0.8]]
    )
    target = torch.tensor([[[0.10, 0.10]], [[0.40, 0.40]]])
    prediction = torch.tensor([[[0.12, 0.12]], [[0.48, 0.48]]])
    visibility = torch.tensor([[2], [2]])
    loss = roi_relative_pose_l1(prediction, target, visibility, boxes)
    assert loss.item() == pytest.approx(0.1, abs=1e-6)
    perfect = roi_relative_pose_l1(target, target, visibility, boxes)
    assert perfect.item() == pytest.approx(0.0, abs=1e-7)


def test_pose_decoder_joint_reference_prevents_joint_collapse():
    decoder = MultiPersonPoseDecoder(
        hidden_size=32, visual_hidden_size=32, decoder_dim=16, num_heads=4
    )
    xy, _ = decoder(
        torch.zeros(1, 32),
        torch.zeros(1, 32),
        torch.zeros(1, 9, 32),
        person_boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        use_joint_reference=True,
    )
    assert xy.std(dim=1).mean().item() > 0.1


def test_topdown_heatmap_pose_decoder_outputs_roi_bound_coordinates():
    torch.manual_seed(11)
    decoder = TopDownHeatmapPoseDecoder(
        hidden_size=32,
        visual_hidden_size=32,
        decoder_dim=16,
        heatmap_size=56,
    )
    assert decoder.final_layer.weight.std().item() < 0.01
    assert torch.count_nonzero(decoder.final_layer.bias).item() == 0
    pose_states = torch.randn(2, 32, requires_grad=True)
    parent_states = torch.randn(2, 32, requires_grad=True)
    roi_tokens = torch.randn(2, 49, 32, requires_grad=True)
    boxes = torch.tensor(
        [[0.1, 0.2, 0.5, 0.9], [0.6, 0.1, 0.95, 0.8]],
        dtype=torch.float32,
    )
    xy, visibility, heatmaps = decoder(
        pose_states,
        parent_states,
        roi_tokens,
        person_boxes=boxes,
    )
    assert xy.shape == (2, 17, 2)
    assert visibility.shape == (2, 17, 3)
    assert heatmaps.shape == (2, 17, 56, 56)
    assert torch.all(xy[:, :, 0] >= boxes[:, None, 0])
    assert torch.all(xy[:, :, 0] <= boxes[:, None, 2])
    assert torch.all(xy[:, :, 1] >= boxes[:, None, 1])
    assert torch.all(xy[:, :, 1] <= boxes[:, None, 3])
    (xy.sum() + visibility.sum() + heatmaps.square().mean()).backward()
    assert roi_tokens.grad is not None
    assert torch.isfinite(roi_tokens.grad).all()


def test_keypoint_heatmap_mse_is_zero_for_exact_gaussian_target():
    boxes = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    target_xy = torch.tensor([[[0.75, 0.25]]])
    visibility = torch.tensor([[2]])
    blank = torch.zeros((1, 1, 8, 8))
    target_loss, target = keypoint_heatmap_mse_loss(
        blank,
        target_xy,
        visibility,
        boxes,
        sigma=1.5,
        return_target=True,
    )
    exact_loss = keypoint_heatmap_mse_loss(
        target,
        target_xy,
        visibility,
        boxes,
        sigma=1.5,
    )
    wrong_loss = keypoint_heatmap_mse_loss(
        target.flip(-1),
        target_xy,
        visibility,
        boxes,
        sigma=1.5,
    )
    assert target_loss > 0
    assert exact_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert wrong_loss > exact_loss
