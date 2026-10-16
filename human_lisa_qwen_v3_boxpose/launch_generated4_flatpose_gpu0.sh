#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
CKPT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/gates/overfit4_query25_300steps_20260723/final-step-300
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/gates/overfit4_query25_300steps_20260723/generated_eval_train4
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 3
fi
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_boxpose/eval_stage1_v3.py   --checkpoint "$CKPT"   --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json   --image-root /root/autodl-tmp/coco/train2017   --output-dir "$OUT" --max-samples 4 --batch-size 2 --max-new-tokens 5000   2>&1 | tee "$OUT.run.log"
