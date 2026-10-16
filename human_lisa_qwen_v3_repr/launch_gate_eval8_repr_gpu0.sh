#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/humanslot
PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/gates/overfit8_boxfree_multiscale_500steps_20260725
CKPT="$RUN/final-step-500"
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json
IMAGES=/root/autodl-tmp/coco/train2017
TF_OUT="$RUN/teacher_forced_metrics.json"
GEN_OUT="$RUN/generated_train8"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
printf '[%s] teacher-forced eval start\n' "$(date '+%F %T %Z')" | tee "$RUN.gate_eval.log"
"$PYTHON" -u human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
  --checkpoint "$CKPT" \
  --annotation "$ANN" \
  --image-root "$IMAGES" \
  --output "$TF_OUT" \
  --max-samples 8 \
  --batch-size 8 \
  2>&1 | tee "$RUN.teacher_forced.log"

printf '[%s] generated eval start\n' "$(date '+%F %T %Z')" | tee -a "$RUN.gate_eval.log"
"$PYTHON" -u human_lisa_qwen_v3_repr/eval_stage1_v3.py \
  --checkpoint "$CKPT" \
  --annotation "$ANN" \
  --image-root "$IMAGES" \
  --output-dir "$GEN_OUT" \
  --max-samples 8 \
  --batch-size 2 \
  --max-new-tokens 5000 \
  2>&1 | tee "$RUN.generated.log"

printf '[%s] gate evaluations completed\n' "$(date '+%F %T %Z')" | tee -a "$RUN.gate_eval.log"
