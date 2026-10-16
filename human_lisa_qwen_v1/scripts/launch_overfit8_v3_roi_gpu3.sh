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
RUN="$ROOT/overfit8_v3_roi_relativepose_300steps_gpu3"
if [ -e "$RUN/train/final-step-300/manifest.json" ]; then
  echo "refusing to overwrite completed v3 run" >&2
  exit 3
fi
mkdir -p "$RUN"
{
  echo "started_at=$(date -Is)"
  echo "architecture=variable_records_roi_relative_pose"
  echo "gpu=3"
} > "$RUN/status.log"

"$PY" -u human_lisa_qwen_v1/train_stage1_v1.py \
  --output-dir "$RUN/train" \
  --max-samples 8 --epochs 150 --max-optimizer-steps 300 \
  --batch-size 1 --grad-accum-steps 4 --max-length 5000 \
  --num-workers 0 --horizontal-flip-prob 0 \
  --visual-lr 5e-5 --language-lr 2e-4 --head-lr 1e-3 \
  --scheduler-type constant --warmup-ratio 0.02 --grad-clip 5 \
  --lora-dropout 0 --log-every 10 --save-every 0 \
  --probe-every 100 --probe-count 4 --probe-max-new-tokens 5000 \
  --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/train.log"

echo "finished_at=$(date -Is)" >> "$RUN/status.log"
echo "status=completed" >> "$RUN/status.log"
