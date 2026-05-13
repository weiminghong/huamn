from __future__ import annotations

import argparse

from human_slot_memory_pipeline.human_slot_memory.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage-2 dense human student entry point.")
    parser.add_argument("--config", default="human_slot_memory_pipeline/configs/stage2_dense_student.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg.ensure_output_dir()
    print("Stage-2 dense student scaffold is ready.")
    print({"dinov3_model": cfg.paths.get("dinov3_model"), "output_dir": str(cfg.output_dir())})


if __name__ == "__main__":
    main()
