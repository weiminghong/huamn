#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
/home/weiminghong/yes/envs/human_v2/bin/python -u human_lisa_qwen_v1/eval_lisa_qwen_v1.py \
  --jsonl /home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/val.jsonl \
  --image-root /share2/program/weiminghong/coco/val2017 \
  --model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --checkpoint /share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/train/final-step-16029 \
  --output-dir /share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/eval_val_legacy_matched_final_gpu1 \
  --max-new-tokens 5000 \
  --batch-size 1 \
  --cuda-visible-devices 1 \
  --iou-thr 0.5 \
  --bf16 \
  --no-allow-seg-token
