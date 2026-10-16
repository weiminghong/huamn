import torch
import torch.nn as nn

from human_lisa_qwen_v4.token_rows_v4 import SelectiveTokenRows


class _TinyTiedModel(nn.Module):
    def __init__(self, vocab=9, hidden=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab, hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.lm_head.weight = self.embedding.weight

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids):
        return self.lm_head(self.embedding(input_ids))


def test_selective_rows_are_the_only_trainable_vocabulary_parameters():
    model = _TinyTiedModel()
    model.requires_grad_(False)
    adapter = SelectiveTokenRows([7, 8], hidden_size=4)
    adapter.attach(model)
    logits = model(torch.tensor([[1, 7, 2, 8]]))
    loss = logits[..., 7].mean() - logits[..., 8].mean()
    loss.backward()
    assert adapter.trainable_parameter_count == 8
    assert adapter.delta.grad is not None
    assert torch.isfinite(adapter.delta.grad).all()
    assert model.embedding.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_detach_restores_original_forward():
    model = _TinyTiedModel()
    model.requires_grad_(False)
    ids = torch.tensor([[7, 2]])
    baseline = model(ids)
    adapter = SelectiveTokenRows([7, 8], hidden_size=4)
    adapter.attach(model)
    with torch.no_grad():
        adapter.delta.fill_(0.5)
    changed = model(ids)
    assert not torch.allclose(changed, baseline)
    adapter.detach()
    restored = model(ids)
    assert torch.allclose(restored, baseline)


def test_semantic_initialization_matches_source_rows_without_training_base_vocab():
    model = _TinyTiedModel()
    model.requires_grad_(False)
    adapter = SelectiveTokenRows([7, 8], hidden_size=4)
    audit = adapter.initialize_from_source_ids(model, [[1], [2, 3]])
    effective = model.embedding.weight.detach()[[7, 8]] + adapter.delta.detach()
    expected = torch.stack(
        [
            model.embedding.weight.detach()[1],
            model.embedding.weight.detach()[[2, 3]].mean(dim=0),
        ]
    )
    assert torch.allclose(effective, expected)
    assert [row["target_token_id"] for row in audit] == [7, 8]
    assert model.embedding.weight.grad is None
    assert model.lm_head.weight.grad is None
