#!/usr/bin/env bash
set -euo pipefail

cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PY=/home/weiminghong/yes/envs/human_v2/bin/python
ROOT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711
RESUME="$ROOT/overfit8_gpu3_100steps/train/final-step-100"
OUT="$ROOT/overfit8_gpu3_500steps_resume"
if [ -e "$OUT/train/final-step-500" ]; then
  echo "refusing to overwrite completed overfit output" >&2
  exit 2
fi
mkdir -p "$OUT"

{
  echo "started_at=$(date -Is)"
  echo "gpu=3"
  echo "resume=$RESUME"
  echo "samples=8"
  echo "target_optimizer_steps=500"
  echo "batch_size=1"
  echo "grad_accum_steps=4"
  echo "max_length=5000"
} > "$OUT/status.log"

"$PY" -u human_lisa_qwen_v1/train_stage1_v1.py \
  --resume-dir "$RESUME" \
  --output-dir "$OUT/train" \
  --max-samples 8 \
  --epochs 250 \
  --max-optimizer-steps 500 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 5000 \
  --num-workers 0 \
  --horizontal-flip-prob 0 \
  --log-every 10 \
  --save-every 0 \
  --probe-every 200 \
  --probe-count 4 \
  --probe-max-new-tokens 5000 \
  --bf16 \
  --cuda-visible-devices 3 \
  2>&1 | tee "$OUT/train.log"
echo "finished_at=$(date -Is)" >> "$OUT/status.log"
echo "status=completed" >> "$OUT/status.log"

