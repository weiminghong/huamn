#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/exact_resume_v1_20260805
OUT="$BASE/independent2"
cd /root/autodl-tmp/humanslot
echo "START $(date -Is)" > "$BASE/independent2.status.log"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 \
  --output-dir "$OUT" \
  --train-ann /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json \
  --train-image-root /root/autodl-tmp/coco/train2017 \
  --model /root/autodl-tmp/model/Qwen3.5-4B \
  --profile vision_tuned --max-samples 80 --epochs 1 \
  --max-optimizer-steps 5 --batch-size 4 --grad-accum-steps 4 \
  --num-workers 0 --max-length 5000 --decoder-warmup-steps 2 \
  --route-loss-hold-steps 1 --route-loss-warmup-steps 3 \
  --save-every 3 --log-every 1 --seed 42 \
  > "$BASE/independent2.log" 2>&1
echo "TRAIN_DONE $(date -Is)" >> "$BASE/independent2.status.log"
set +e
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.compare_checkpoints_v4 \
  --first "$BASE/continuous/epoch-1-step-5" \
  --second "$OUT/epoch-1-step-5" \
  --output "$BASE/comparison_independent.json" \
  > "$BASE/comparison_independent.log" 2>&1
code=$?
set -e
echo "COMPARE_EXIT=$code $(date -Is)" >> "$BASE/independent2.status.log"
