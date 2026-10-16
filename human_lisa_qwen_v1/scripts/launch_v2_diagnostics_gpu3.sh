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
RUN="$ROOT/overfit8_v2_balanced_300steps_gpu3"
CKPT="$RUN/train/final-step-300"
ANN=/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json
IMAGES=/share2/program/weiminghong/coco/train2017

test -f "$CKPT/manifest.json"
for name in eval_train8_official teacher_forced_train8 oracle_head_train8; do
  if [ -e "$RUN/$name/metrics.json" ]; then
    echo "refusing to overwrite $RUN/$name/metrics.json" >&2
    exit 3
  fi
  mkdir -p "$RUN/$name"
done

"$PY" -u human_lisa_qwen_v1/eval_stage1_v1.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output-dir "$RUN/eval_train8_official" --max-samples 8 --batch-size 1 \
  --max-new-tokens 5000 --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/eval_train8_official/eval.log"

"$PY" -u human_lisa_qwen_v1/teacher_forced_audit_v1.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output "$RUN/teacher_forced_train8/metrics.json" --max-samples 8 \
  --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/teacher_forced_train8/audit.log"

"$PY" -u human_lisa_qwen_v1/oracle_head_eval_v1.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output "$RUN/oracle_head_train8/metrics.json" --max-samples 8 \
  --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/oracle_head_train8/eval.log"

echo "status=completed"
