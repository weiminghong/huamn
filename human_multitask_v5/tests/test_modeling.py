import torch

from human_multitask_v5.losses import parsing_loss, segment_loss
from human_multitask_v5.modeling import PersonConditionedMask2Former


def test_mask2former_interface_shapes_and_gradients():
    torch.manual_seed(3)
    model = PersonConditionedMask2Former(
        text_dim=24,
        qwen_feature_channels=[16, 16, 16],
        hidden_dim=32,
        num_parts=3,
        encoder_layers=1,
        decoder_layers=1,
        attention_heads=4,
    )
    maps = [
        torch.randn(1, 16, 4, 4, requires_grad=True)
        for _ in range(3)
    ]
    output = model(
        layer_maps=maps,
        owner_states=torch.randn(2, 24, requires_grad=True),
        boxes_normalized=torch.tensor(
            [[0.1, 0.1, 0.5, 0.9], [0.4, 0.2, 0.8, 0.95]]
        ),
        segment_owner_indices=torch.tensor([0]),
        parsing_owner_indices=torch.tensor([1]),
    )
    assert output.segment_logits.shape[:1] == (1,)
    assert output.parsing_logits.shape[:2] == (1, 3)
    segment = segment_loss(
        output.segment_logits,
        torch.zeros_like(output.segment_logits),
    )["total"]
    labels = torch.zeros(
        (1, *output.parsing_logits.shape[-2:]), dtype=torch.long
    )
    labels[:, 1:3, 1:3] = 1
    parsing = parsing_loss(output.parsing_logits, labels)["total"]
    (segment + parsing).backward()
    assert all(value.grad is not None for value in maps)
