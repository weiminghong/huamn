#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/human4b/bin/python
CKPT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/stage1_v1_qwen35_4b_5task_3epoch_epochval_20260718_223143_gpu0/train/final-step-5343
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
IMG=/root/autodl-tmp/coco/val2017
OUTDIR=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/diagnostics_epoch1_teacher_forced_head_20260722
mkdir -p "$OUTDIR"
"$PY" -u human_lisa_qwen_v1/oracle_head_eval_v1.py \
  --checkpoint "$CKPT" \
  --annotation "$ANN" \
  --image-root "$IMG" \
  --max-samples 2693 \
  --cuda-visible-devices 0 \
  --output "$OUTDIR/full2693_metrics.json" \
  2>&1 | tee "$OUTDIR/full2693.log"
