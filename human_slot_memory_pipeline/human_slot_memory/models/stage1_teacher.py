from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn


class StructuredHumanTeacher(nn.Module):
    def __init__(self, qwen_model_path: str, dry_run: bool = False) -> None:
        super().__init__()
        self.qwen_model_path = qwen_model_path
        self.dry_run = dry_run
        self.model: Optional[nn.Module] = None
        self.processor: Any = None
        if not dry_run:
            self._load_model()

    def _load_model(self) -> None:
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.qwen_model_path, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(self.qwen_model_path, trust_remote_code=True)

    def forward(self, input_ids: torch.Tensor | None = None, **kwargs: Any) -> dict[str, torch.Tensor]:
        if self.dry_run or self.model is None:
            batch_size = 1 if input_ids is None else input_ids.shape[0]
            return {
                "text_loss": torch.zeros((), device=input_ids.device if input_ids is not None else None),
                "slot_embeddings": torch.zeros(batch_size, 32, 256),
            }
        outputs = self.model(input_ids=input_ids, output_hidden_states=True, **kwargs)
        return {"text_loss": outputs.loss, "hidden_states": outputs.hidden_states}
