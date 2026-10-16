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
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v4_topdown_heatmap_100steps_gpu3
CKPT="$RUN/train/final-step-100"
OUT="$RUN/eval_train8_free"
test -f "$CKPT/manifest.json"
test ! -e "$OUT/metrics.json"
mkdir -p "$OUT"
{
  echo "started_at=$(date -Is)"
  echo "checkpoint=$CKPT"
  echo "gpu=3"
} > "$OUT/status.log"
finish() {
  code=$?
  trap - EXIT
  echo "finished_at=$(date -Is)" >> "$OUT/status.log"
  echo "exit_code=$code" >> "$OUT/status.log"
  if [ "$code" -eq 0 ]; then
    echo "status=completed" >> "$OUT/status.log"
  else
    echo "status=failed" >> "$OUT/status.log"
  fi
  exit "$code"
}
trap finish EXIT
"$PY" -u human_lisa_qwen_v1/eval_stage1_v1.py \
  --checkpoint "$CKPT" \
  --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json \
  --image-root /share2/program/weiminghong/coco/train2017 \
  --output-dir "$OUT" --max-samples 8 --batch-size 1 \
  --max-new-tokens 5000 --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$OUT/eval.log"
