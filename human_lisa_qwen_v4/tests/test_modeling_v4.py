from types import SimpleNamespace

import torch
import torch.nn as nn

from human_lisa_qwen_v4.modeling_v4 import Stage1V4Model
from human_lisa_qwen_v4.targets_v4 import TEXT_TASK_TO_ID


class _FakeVisionLanguageModel(nn.Module):
    def __init__(self, vocab=16, hidden=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.config = SimpleNamespace(use_cache=True)

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids,
        attention_mask=None,
        output_hidden_states=False,
        return_dict=True,
        **kwargs,
    ):
        del attention_mask, output_hidden_states, return_dict, kwargs
        hidden = self.embedding(input_ids)
        return SimpleNamespace(
            logits=self.lm_head(hidden),
            hidden_states=[hidden],
        )


def test_v4_forward_backward_is_finite_and_base_stays_frozen():
    base = _FakeVisionLanguageModel()
    base.requires_grad_(False)
    model = Stage1V4Model(
        base_model=base,
        hidden_size=32,
        token_ids={"pose": 14, "seg": 15},
        image_token_id=3,
        visual_merge_size=2,
        decoder_dim=16,
        decoder_heads=4,
        decoder_layers=1,
    )
    input_ids = torch.tensor([[3, 3, 3, 3, 5, 14, 15, 6]])
    labels = torch.tensor([[-100, -100, -100, -100, 5, 14, 15, 6]])
    task_ids = torch.tensor(
        [
            [
                0,
                0,
                0,
                0,
                TEXT_TASK_TO_ID["structure"],
                TEXT_TASK_TO_ID["pose_route"],
                TEXT_TASK_TO_ID["seg_route"],
                TEXT_TASK_TO_ID["structure"],
            ]
        ]
    )
    pose_target = torch.zeros(1, 17, 3)
    pose_target[..., 0] = 0.4
    pose_target[..., 1] = 0.5
    pose_target[..., 2] = 2
    seg_target = torch.zeros(1, 16, 16, dtype=torch.bool)
    seg_target[:, 3:11, 4:12] = True
    outputs = model(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        labels=labels,
        text_task_ids=task_ids,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        person_counts=[1],
        pose_targets=[pose_target],
        pose_areas=[torch.tensor([100.0])],
        seg_targets=[seg_target],
        pose_person_indices=[torch.tensor([0])],
        seg_person_indices=[torch.tensor([0])],
        original_sizes=[(16, 16)],
        return_predictions=True,
    )
    assert torch.isfinite(outputs["loss"])
    outputs["loss"].backward()
    assert model.token_rows.delta.grad is not None
    assert torch.isfinite(model.token_rows.delta.grad).all()
    assert any(
        parameter.grad is not None for parameter in model.pose_decoder.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.mask_decoder.parameters()
    )
    assert base.embedding.weight.grad is None
    assert outputs["predictions"][0]["pose_coordinates"].shape == (1, 17, 2)
    assert outputs["predictions"][0]["mask_logits"].shape[0] == 1
