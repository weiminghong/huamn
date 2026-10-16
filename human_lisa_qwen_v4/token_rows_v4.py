from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveTokenRows(nn.Module):
    """Train only additive rows for newly added tied input/output tokens."""

    def __init__(self, token_ids: Sequence[int], hidden_size: int) -> None:
        super().__init__()
        ids = [int(value) for value in token_ids]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("special token ids must be non-empty and unique")
        self.register_buffer(
            "token_ids", torch.tensor(ids, dtype=torch.long), persistent=True
        )
        self.delta = nn.Parameter(torch.zeros(len(ids), int(hidden_size)))
        self._handles: List[torch.utils.hooks.RemovableHandle] = []

    @property
    def trainable_parameter_count(self) -> int:
        return int(self.delta.numel())

    def initialize_from_source_ids(
        self,
        model: nn.Module,
        source_token_ids: Sequence[Sequence[int]],
    ) -> List[dict]:
        """Initialize each effective special row from ordinary semantic tokens."""
        if len(source_token_ids) != len(self.token_ids):
            raise ValueError("one source-token group is required per special token")
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise RuntimeError("model must expose input and output embeddings")
        if input_embeddings.weight.data_ptr() != output_embeddings.weight.data_ptr():
            raise RuntimeError("selective shared rows require tied input/output weights")

        base = input_embeddings.weight.detach().float()
        rows = []
        audits: List[dict] = []
        for target_id, source_ids in zip(
            self.token_ids.tolist(), source_token_ids
        ):
            sources = [int(value) for value in source_ids]
            if not sources:
                raise ValueError("semantic token initialization cannot be empty")
            if target_id < 0 or target_id >= int(base.shape[0]):
                raise ValueError(f"special token id is outside embedding table: {target_id}")
            if min(sources) < 0 or max(sources) >= int(base.shape[0]):
                raise ValueError(f"source token ids are outside embedding table: {sources}")
            source = base[sources].mean(dim=0)
            delta = source - base[target_id]
            rows.append(delta)
            audits.append(
                {
                    "target_token_id": int(target_id),
                    "source_token_ids": sources,
                    "base_norm": float(base[target_id].norm().item()),
                    "source_norm": float(source.norm().item()),
                    "delta_norm": float(delta.norm().item()),
                }
            )
        with torch.no_grad():
            self.delta.copy_(
                torch.stack(rows).to(
                    device=self.delta.device, dtype=self.delta.dtype
                )
            )
        return audits

    def attach(self, model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("selective token rows are already attached")
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is None or output_embeddings is None:
            raise RuntimeError("model must expose input and output embeddings")
        if int(input_embeddings.embedding_dim) != int(self.delta.shape[1]):
            raise ValueError("input embedding dimension does not match token rows")

        def add_input_delta(
            module: nn.Module, args: tuple, output: torch.Tensor
        ) -> torch.Tensor:
            del module
            input_ids = args[0]
            delta = self.delta.to(device=output.device, dtype=output.dtype)
            addition = output.new_zeros(output.shape)
            for row, token_id in enumerate(self.token_ids.tolist()):
                addition = addition + input_ids.eq(token_id).unsqueeze(-1) * delta[row]
            return output + addition

        def add_output_delta(
            module: nn.Module, args: tuple, output: torch.Tensor
        ) -> torch.Tensor:
            del module
            hidden = args[0]
            delta = self.delta.to(device=hidden.device, dtype=hidden.dtype)
            special_logits = F.linear(hidden, delta).to(output.dtype)
            token_ids = self.token_ids.to(device=output.device)
            return torch.index_add(output, -1, token_ids, special_logits)

        self._handles = [
            input_embeddings.register_forward_hook(add_input_delta),
            output_embeddings.register_forward_hook(add_output_delta),
        ]

    def detach(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def extra_repr(self) -> str:
        return (
            f"tokens={self.token_ids.tolist()}, "
            f"trainable={self.trainable_parameter_count}"
        )


def add_v4_special_tokens(processor, model) -> dict[str, int]:
    from .schema_v4 import SPECIAL_TOKENS_V4

    tokenizer = processor.tokenizer
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    additions = [token for token in SPECIAL_TOKENS_V4 if token not in existing]
    if additions:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + additions}
        )
        if len(tokenizer) > int(model.get_input_embeddings().num_embeddings):
            model.resize_token_embeddings(len(tokenizer))
    token_ids = {
        name: int(tokenizer.convert_tokens_to_ids(token))
        for name, token in zip(("pose", "seg"), SPECIAL_TOKENS_V4)
    }
    if len(set(token_ids.values())) != 2 or any(value < 0 for value in token_ids.values()):
        raise RuntimeError(f"invalid v4 special token ids: {token_ids}")
    return token_ids
