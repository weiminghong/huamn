#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_boxpose/train_stage1_v3.py \
  --output-dir /root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/overfit8_boxref_100_gpu0_20260724b \
  --run-kind overfit --max-samples 8 --epochs 200 --max-optimizer-steps 100 \
  --batch-size 4 --grad-accum-steps 1 --max-length 5000 --num-workers 4 \
  --horizontal-flip-prob 0 --max-people 25 --person-query-layers 3 --pose-decoder-layers 2 \
  --lora-target-mode all-linear --scheduler-type constant --warmup-ratio 0.02 \
  --visual-lr 1e-5 --language-lr 1e-5 --token-lr 2e-5 --head-lr 1e-3 \
  --weight-decay 0.01 --grad-clip 5 \
  --query-objectness-weight 1 --query-bbox-l1-weight 5 --query-giou-weight 2 --query-aux-weight 0.5 \
  --pose-l1-weight 10 --pose-oks-weight 4 --pose-visibility-weight 1 --pose-aux-weight 0.5 \
  --mask-bce-weight 2 --mask-dice-weight 5 --log-every 10 --no-save-epoch-checkpoints \
  2>&1 | tee /root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/overfit8_boxref_100_gpu0_20260724b.train.log
