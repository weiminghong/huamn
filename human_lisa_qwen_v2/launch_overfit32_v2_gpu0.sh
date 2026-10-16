#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v2/gates/overfit32_100steps_20260723
LOG=${OUT}.log
mkdir -p "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/human4b/bin/python human_lisa_qwen_v2/train_stage1_v2.py \
  --output-dir "$OUT" --run-kind overfit --max-samples 32 --epochs 13 \
  --max-optimizer-steps 100 --batch-size 4 --grad-accum-steps 1 \
  --num-workers 4 --skip-save --log-every 5 >"$LOG" 2>&1
