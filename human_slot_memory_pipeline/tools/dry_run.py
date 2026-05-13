from __future__ import annotations

import argparse

import torch

from human_slot_memory_pipeline.human_slot_memory.config import load_config
from human_slot_memory_pipeline.human_slot_memory.data import build_dataset, human_collate
from human_slot_memory_pipeline.human_slot_memory.losses import PartialLabelLoss
from human_slot_memory_pipeline.human_slot_memory.models import DenseHumanStudent, HumanMemoryProjector, MemoryLanguageReasoner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a lightweight Human Slot Memory wiring check.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg.ensure_output_dir()
    val_ann = str(cfg.paths.get("val_ann_json") or cfg.paths.get("val_jsonl"))
    val_root = str(cfg.paths.get("val_image_root") or cfg.paths.get("image_root"))
    dataset = build_dataset(val_ann, image_root=val_root, max_samples=args.max_samples)
    batch = human_collate([dataset[0]])
    hidden_dim = int(cfg.stage2.get("hidden_dim", 256))
    max_queries = int(cfg.stage2.get("max_queries", 32))
    student = DenseHumanStudent(
        dinov3_model_path=str(cfg.paths.get("dinov3_model", "")),
        hidden_dim=hidden_dim,
        max_queries=max_queries,
        dry_run=True,
    )
    pixel_values = torch.zeros(batch.batch_size, 3, 224, 224)
    outputs = student(pixel_values)
    losses = PartialLabelLoss()(outputs, batch)
    memory = HumanMemoryProjector(
        dense_dim=hidden_dim,
        memory_dim=hidden_dim,
        memory_tokens=int(cfg.stage3.get("memory_tokens", max_queries)),
    )(outputs["queries"], outputs["validity_logits"])
    reasoner = MemoryLanguageReasoner(
        qwen_model_path=str(cfg.paths.get("qwen_model", "")),
        memory_dim=hidden_dim,
        qwen_dim=hidden_dim,
        dry_run=True,
    )
    reasoner_out = reasoner(memory)
    print(
        {
            "samples": batch.batch_size,
            "slots": len(batch.samples[0].slots),
            "queries": list(outputs["queries"].shape),
            "memory": list(memory.shape),
            "losses": {name: float(value.detach().cpu()) for name, value in losses.items()},
            "stage3_text_loss": float(reasoner_out["text_loss"].detach().cpu()),
        }
    )


if __name__ == "__main__":
    main()
