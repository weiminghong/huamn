from __future__ import annotations

import argparse

from human_slot_memory_pipeline.human_slot_memory.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-3 dense-to-language memory loop entry point.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/stage3_memory_loop.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.ensure_output_dir()
    print("Stage-3 memory loop scaffold is ready.")
    print({"qwen_model": cfg.paths.get("qwen_model"), "output_dir": str(cfg.output_dir())})


if __name__ == "__main__":
    main()
