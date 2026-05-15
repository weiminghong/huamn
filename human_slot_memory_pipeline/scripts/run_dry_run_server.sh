#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/weiminghong/humanslot}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}"

"${PYTHON_BIN}" -m human_slot_memory_pipeline.tools.dry_run \
  --config human_slot_memory_pipeline/configs/pipeline_debug.yaml \
  --max-samples 1
