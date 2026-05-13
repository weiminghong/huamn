from __future__ import annotations

import argparse

from human_slot_memory_pipeline.human_slot_memory.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Slot Memory evaluation scaffold.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/pipeline_debug.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print("Evaluation scaffold is ready.")
    print({"val_jsonl": cfg.paths.get("val_jsonl"), "output_dir": str(cfg.output_dir())})


if __name__ == "__main__":
    main()
