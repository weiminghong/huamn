from __future__ import annotations

import argparse

from human_slot_memory_pipeline.human_slot_memory.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Human Slot Memory inference scaffold.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/pipeline_debug.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print("Inference scaffold is ready.")
    print({"dinov3_model": cfg.paths.get("dinov3_model"), "qwen_model": cfg.paths.get("qwen_model")})


if __name__ == "__main__":
    main()
