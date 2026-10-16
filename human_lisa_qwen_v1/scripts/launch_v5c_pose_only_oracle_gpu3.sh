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
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v5c_pose_only_resume500_to800_gpu3
CKPT="$RUN/train/final-step-800"
OUT="$RUN/oracle_head_train8"
test -f "$CKPT/manifest.json"
test ! -e "$OUT/metrics.json"
mkdir -p "$OUT"
{
  echo "started_at=$(date -Is)"
  echo "checkpoint=$CKPT"
  echo "protocol=oracle_pose_only"
  echo "gpu=3"
} > "$RUN/eval_status.log"
finish() {
  code=$?
  trap - EXIT
  echo "finished_at=$(date -Is)" >> "$RUN/eval_status.log"
  echo "exit_code=$code" >> "$RUN/eval_status.log"
  if [ "$code" -eq 0 ]; then
    echo "status=completed" >> "$RUN/eval_status.log"
  else
    echo "status=failed" >> "$RUN/eval_status.log"
  fi
  exit "$code"
}
trap finish EXIT
"$PY" -u human_lisa_qwen_v1/oracle_head_eval_v1.py \
  --checkpoint "$CKPT" \
  --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json \
  --image-root /share2/program/weiminghong/coco/train2017 \
  --output "$OUT/metrics.json" --max-samples 8 --bf16 \
  --cuda-visible-devices 3 2>&1 | tee "$OUT/eval.log"

