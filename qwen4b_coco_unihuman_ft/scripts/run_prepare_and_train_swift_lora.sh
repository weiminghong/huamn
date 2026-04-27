#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAX_PIXELS=${MAX_PIXELS:-1003520}
export MODEL_PATH=${MODEL_PATH:-/home/priv/project/model/Qwen2.5-VL-3B-Instruct}

bash qwen4b_coco_unihuman_ft/scripts/run_prepare_server_data.sh
bash qwen4b_coco_unihuman_ft/scripts/run_train_swift_lora.sh
