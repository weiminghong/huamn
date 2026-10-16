#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/human4b/bin/python
OUT=/root/autodl-tmp/humanslot_outputs/qwen35_zero_shot_box_attr_v1/batch4check8_20260722
mkdir -p "$OUT"
"$PY" -u qwen4b_coco_unihuman_ft/eval_zero_shot_box_attr_qwen35.py \
  --model /root/autodl-tmp/model/Qwen3.5-4B \
  --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
  --image-root /root/autodl-tmp/coco/val2017 \
  --output-dir "$OUT" \
  --max-samples 8 \
  --batch-size 4 \
  --max-new-tokens 1024 \
  --cuda-visible-devices 0 \
  2>&1 | tee "$OUT/run.log"
