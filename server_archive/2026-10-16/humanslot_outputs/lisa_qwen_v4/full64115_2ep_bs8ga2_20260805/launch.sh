#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
mkdir -p /root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/full64115_2ep_bs8ga2_20260805/run
exec /root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 \
  --train-ann /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json \
  --train-image-root /root/autodl-tmp/coco/train2017 \
  --model /root/autodl-tmp/model/Qwen3.5-4B \
  --output-dir /root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/full64115_2ep_bs8ga2_20260805/run \
  --profile vision_tuned \
  --epochs 2 \
  --batch-size 8 \
  --grad-accum-steps 2 \
  --num-workers 8 \
  --max-length 5000 \
  --save-every 500 \
  --log-every 10 \
  --token-init semantic \
  --decoder-warmup-steps 200 \
  --visual-warmup-steps 200 \
  --seed 42 \
  --deterministic
