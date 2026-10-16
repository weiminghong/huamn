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
CKPT="$ROOT/overfit8_gpu3_500steps_resume/train/final-step-500"
OUT="$ROOT/overfit8_gpu3_500steps_resume/teacher_forced_train8"
if [ ! -f "$CKPT/manifest.json" ]; then
  echo "missing final checkpoint: $CKPT" >&2
  exit 2
fi
mkdir -p "$OUT"
"$PY" -u human_lisa_qwen_v1/teacher_forced_audit_v1.py   --checkpoint "$CKPT"   --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json   --image-root /share2/program/weiminghong/coco/train2017   --output "$OUT/metrics.json"   --max-samples 8   --bf16   --cuda-visible-devices 3   2>&1 | tee "$OUT/audit.log"
