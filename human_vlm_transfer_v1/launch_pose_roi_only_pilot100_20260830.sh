#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/autodl-tmp/humanslot
cd /root/autodl-tmp/humanslot

OUT=/root/autodl-tmp/humanslot_outputs/stage1_transfer_v1_pose_roi_only_pilot100_20260830
LOG=/root/autodl-tmp/humanslot_outputs/stage1_transfer_v1_pose_roi_only_pilot100_20260830.launch.log
EXIT_FILE=/root/autodl-tmp/humanslot_outputs/stage1_transfer_v1_pose_roi_only_pilot100_20260830.exit

set +e
/root/miniconda3/envs/human4b/bin/python -m human_vlm_transfer_v1.train_stage1_pilot \
  --output-dir "$OUT" \
  --max-samples 512 \
  --steps 100 \
  --batch-size 1 \
  --num-workers 0 \
  --select-task-rich \
  --vision-start-block 16 \
  --vision-layers 7,15,23 \
  --num-queries 100 \
  --decoder-dim 256 \
  --decoder-layers 3 \
  --decoder-heads 8 \
  --save-compact \
  >"$LOG" 2>&1
code=$?
printf '%s\n' "$code" >"$EXIT_FILE"
exit "$code"
