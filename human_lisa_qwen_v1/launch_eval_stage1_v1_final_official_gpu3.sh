#!/usr/bin/env bash
set -euo pipefail

REPO=/home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_v2/bin/python
RUN_ROOT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3
TRAIN_DIR="$RUN_ROOT/train"
OUT_DIR="$RUN_ROOT/eval_val_official_final_gpu3"
LOG="$RUN_ROOT/eval_val_official_final_gpu3.log"

export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=3

cd "$REPO"

if pgrep -f 'human_lisa_qwen_v1/train_stage1_v1.py.*stage1_v1_full_singlebackward_20260712_gpu3' >/dev/null; then
  echo "[guard] Stage1 training is still running; refusing to start official val eval."
  exit 2
fi

CKPT=$(find "$TRAIN_DIR" -maxdepth 1 -type d -name 'final-step-*' | sort -V | tail -n 1)
if [[ -z "${CKPT:-}" ]]; then
  echo "[guard] No final-step-* checkpoint found under $TRAIN_DIR"
  exit 3
fi

mkdir -p "$OUT_DIR"
echo "[eval] checkpoint=$CKPT"
echo "[eval] output=$OUT_DIR"
"$PY" -u human_lisa_qwen_v1/eval_stage1_v1.py   --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json   --image-root /share2/program/weiminghong/coco/val2017   --checkpoint "$CKPT"   --output-dir "$OUT_DIR"   --max-new-tokens 5000   --batch-size 1   --bf16 2>&1 | tee "$LOG"
