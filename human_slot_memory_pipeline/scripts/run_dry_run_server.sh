#!/usr/bin/env bash
set -euo pipefail

cd /home/priv/wmh_project/human
export PYTHONPATH=/home/priv/wmh_project/human

/home/priv/miniconda3/envs/human_v2/bin/python -m human_slot_memory_pipeline.tools.dry_run \
  --config human_slot_memory_pipeline/configs/pipeline_debug.yaml \
  --max-samples 1
