#!/usr/bin/env bash
set -euo pipefail

REPO=/home/weiminghong/humanslot
PYTHON=/home/weiminghong/yes/envs/human_v2/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v5d_strict_pose_only_resetopt_300steps_gpu3
EVAL_DIR="$RUN/oracle_head_train8"
mkdir -p "$EVAL_DIR"
cd "$REPO"
CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u human_lisa_qwen_v1/oracle_head_eval_v1.py \
  --checkpoint "$RUN/train/final-step-300" \
  --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json \
  --image-root /share2/program/weiminghong/coco/train2017 \
  --output "$EVAL_DIR/metrics.json" \
  --max-samples 8 \
  --cuda-visible-devices 3 \
  --bf16 \
  > "$EVAL_DIR/eval.log" 2>&1
