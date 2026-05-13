from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class MemoryLanguageReasoner(nn.Module):
    def __init__(self, qwen_model_path: str, memory_dim: int = 256, qwen_dim: int = 256, dry_run: bool = False) -> None:
        super().__init__()
        self.qwen_model_path = qwen_model_path
        self.dry_run = dry_run
        self.memory_to_qwen = nn.Linear(memory_dim, qwen_dim)
        self.model: Optional[nn.Module] = None
        self.processor: Any = None
        if not dry_run:
            self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.qwen_model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.qwen_model_path, trust_remote_code=True)

    def forward(self, memory_tokens: torch.Tensor, input_ids: torch.Tensor | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        projected_memory = self.memory_to_qwen(memory_tokens)
        if self.dry_run or self.model is None:
            return {"text_loss": projected_memory.mean() * 0.0, "projected_memory": projected_memory}
        outputs = self.model(input_ids=input_ids, **kwargs)
        return {"text_loss": outputs.loss, "projected_memory": projected_memory}
