import torch

from human_lisa_qwen_v3_repr.heads_v3 import QueryConditionedPoseDecoder


def _inputs():
    torch.manual_seed(7)
    people = 3
    query = torch.randn(people, 32)
    person = torch.randn(people, 48)
    pose = torch.randn(people, 48)
    features = [
        torch.randn(1, 32, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 32, 2, 2),
    ]
    boxes_a = torch.tensor(
        [[0.2, 0.2, 0.1, 0.1], [0.5, 0.5, 0.2, 0.3], [0.8, 0.7, 0.1, 0.2]],
        dtype=torch.float32,
    )
    boxes_b = torch.tensor(
        [[0.9, 0.9, 0.8, 0.8], [0.1, 0.8, 0.7, 0.1], [0.4, 0.2, 0.5, 0.6]],
        dtype=torch.float32,
    )
    return query, person, pose, features, boxes_a, boxes_b


def test_pose_is_box_invariant():
    decoder = QueryConditionedPoseDecoder(
        text_hidden_size=48, decoder_dim=32, num_heads=4, num_layers=2
    ).eval()
    query, person, pose, features, boxes_a, boxes_b = _inputs()
    xy_a, vis_a, aux_a = decoder(query, person, pose, boxes_a, features)
    xy_b, vis_b, aux_b = decoder(query, person, pose, boxes_b, features)
    torch.testing.assert_close(xy_a, xy_b, rtol=0, atol=0)
    torch.testing.assert_close(vis_a, vis_b, rtol=0, atol=0)
    assert len(aux_a) == len(aux_b) == 1
    assert xy_a.shape == (3, 17, 2)
    assert vis_a.shape == (3, 17, 3)


def test_pose_reads_visual_feature_maps():
    decoder = QueryConditionedPoseDecoder(
        text_hidden_size=48, decoder_dim=32, num_heads=4, num_layers=2
    ).eval()
    query, person, pose, features, boxes_a, _ = _inputs()
    changed = [feature.clone() for feature in features]
    changed[0] = changed[0] + 3.0
    _, vis_a, _ = decoder(query, person, pose, boxes_a, features)
    _, vis_b, _ = decoder(query, person, pose, boxes_a, changed)
    assert not torch.allclose(vis_a, vis_b)


def test_pose_backpropagates_to_visual_features():
    decoder = QueryConditionedPoseDecoder(
        text_hidden_size=48, decoder_dim=32, num_heads=4, num_layers=2
    )
