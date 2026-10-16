#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=3
export PYTHONUNBUFFERED=1
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_stable5task_full_3epoch_20260716_044540_gpu2
CKPT="$RUN/train/checkpoint-10000"
OUT="$RUN/eval_val_full_checkpoint-10000_gpu3"
LOG="$OUT/eval.log"
{
  echo "[start] $(date -Is)"
  echo "checkpoint=$CKPT"
  echo "output=$OUT"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  /home/weiminghong/yes/envs/human_v2/bin/python -u human_lisa_qwen_v1/eval_stage1_v1.py \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --batch-size 1 \
    --max-new-tokens 5000 \
    --cuda-visible-devices 3 \
    --bf16
  echo "[done] $(date -Is)"
} 2>&1 | tee -a "$LOG"
