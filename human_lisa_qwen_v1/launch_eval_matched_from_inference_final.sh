#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export PYTHONPATH=/home/weiminghong/humanslot:${PYTHONPATH:-}
OUT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/eval_val_matched_diagnostic_from_official_inference
mkdir -p "$OUT"
exec >> "$OUT/eval.log" 2>&1
echo "START $(date)"
/home/weiminghong/yes/envs/human_v2/bin/python -u /home/weiminghong/humanslot/human_lisa_qwen_v1/eval_matched_from_inference_v1.py \
  --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json \
  --image-root /share2/program/weiminghong/coco/val2017 \
  --inference /share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/eval_val_official_final_gpu3/inference.jsonl \
  --output "$OUT/metrics.json" \
  --wait \
  --poll-seconds 300 \
  --iou-thr 0.5
echo "DONE $(date)"
