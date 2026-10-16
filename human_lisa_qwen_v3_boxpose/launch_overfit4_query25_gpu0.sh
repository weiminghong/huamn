#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
exec /root/miniconda3/envs/human4b/bin/python human_lisa_qwen_v3/train_stage1_v3.py \
  --output-dir /root/autodl-tmp/humanslot_outputs/lisa_qwen_v3/gates/overfit4_query25_300steps_20260723 \
  --run-kind overfit \
  --max-samples 4 \
  --epochs 300 \
  --max-optimizer-steps 300 \
  --batch-size 4 \
  --grad-accum-steps 1 \
  --num-workers 4 \
  --horizontal-flip-prob 0 \
  --max-people 25 \
  --person-query-layers 3 \
  --pose-decoder-layers 2 \
  --scheduler-type constant \
  --warmup-ratio 0.02 \
  --visual-lr 1e-5 \
  --language-lr 1e-5 \
  --token-lr 2e-5 \
  --head-lr 2e-4 \
  --grad-clip 5 \
  --query-objectness-weight 1 \
  --query-bbox-l1-weight 5 \
  --query-giou-weight 2 \
  --query-aux-weight 0.5 \
  --mask-bce-weight 2 \
  --mask-dice-weight 5 \
  --no-save-epoch-checkpoints \
  --log-every 10
