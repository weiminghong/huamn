#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/exact_resume_v1_20260805
CONTINUOUS="$BASE/continuous"
RESUMED="$BASE/resumed"
STATUS="$BASE/status.log"
cd /root/autodl-tmp/humanslot

echo "START $(date -Is)" > "$STATUS"
COMMON=(
  --train-ann /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json
  --train-image-root /root/autodl-tmp/coco/train2017
  --model /root/autodl-tmp/model/Qwen3.5-4B
  --profile vision_tuned
  --max-samples 80
  --epochs 1
  --max-optimizer-steps 5
  --batch-size 4
  --grad-accum-steps 4
  --num-workers 0
  --max-length 5000
  --decoder-warmup-steps 2
  --route-loss-hold-steps 1
  --route-loss-warmup-steps 3
  --save-every 3
  --log-every 1
  --seed 42
)

echo "CONTINUOUS_START $(date -Is)" >> "$STATUS"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 \
  --output-dir "$CONTINUOUS" "${COMMON[@]}" \
  > "$BASE/continuous.log" 2>&1
echo "CONTINUOUS_DONE $(date -Is)" >> "$STATUS"

echo "RESUME_START $(date -Is)" >> "$STATUS"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 \
  --output-dir "$RESUMED" \
  --resume-from "$CONTINUOUS/step-3" \
  "${COMMON[@]}" \
  > "$BASE/resumed.log" 2>&1
echo "RESUME_DONE $(date -Is)" >> "$STATUS"

/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.compare_checkpoints_v4 \
  --first "$CONTINUOUS/epoch-1-step-5" \
  --second "$RESUMED/epoch-1-step-5" \
  --output "$BASE/comparison.json" \
  > "$BASE/comparison.log" 2>&1

echo "SUCCESS $(date -Is)" >> "$STATUS"
