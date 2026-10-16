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
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v4b_bboxjitter_resume100_to200_gpu3
CKPT="$RUN/train/final-step-200"
ORACLE="$RUN/oracle_head_train8"
FREE="$RUN/eval_train8_free"
test -f "$CKPT/manifest.json"
test ! -e "$ORACLE/metrics.json"
test ! -e "$FREE/metrics.json"
mkdir -p "$ORACLE" "$FREE"
{
  echo "started_at=$(date -Is)"
  echo "checkpoint=$CKPT"
  echo "gpu=3"
  echo "phase=oracle"
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
  --output "$ORACLE/metrics.json" --max-samples 8 --bf16 \
  --cuda-visible-devices 3 2>&1 | tee "$ORACLE/eval.log"
echo "phase=free" >> "$RUN/eval_status.log"
"$PY" -u human_lisa_qwen_v1/eval_stage1_v1.py \
  --checkpoint "$CKPT" \
  --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json \
  --image-root /share2/program/weiminghong/coco/train2017 \
  --output-dir "$FREE" --max-samples 8 --batch-size 1 \
  --max-new-tokens 5000 --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$FREE/eval.log"
