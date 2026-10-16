#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v2/gates/overfit32_v2b_200steps_20260723
LOG=${OUT}.log
mkdir -p "$(dirname "$OUT")"
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/human4b/bin/python human_lisa_qwen_v2/train_stage1_v2.py \
  --output-dir "$OUT" --run-kind overfit --max-samples 32 --epochs 25 \
  --max-optimizer-steps 200 --batch-size 4 --grad-accum-steps 1 --num-workers 4 \
  --horizontal-flip-prob 0 --scheduler-type constant --warmup-ratio 0.02 \
  --visual-lr 1e-5 --language-lr 1e-5 --token-lr 2e-5 --head-lr 2e-4 \
  --grad-clip 5 --mask-bce-weight 2 --mask-dice-weight 5 \
  --no-save-epoch-checkpoints --log-every 10 >"$LOG" 2>&1
