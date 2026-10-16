#!/usr/bin/env bash
set -euo pipefail

REPO=/home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_v2/bin/python
RUN_ROOT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3
INFER="$RUN_ROOT/eval_val_official_final_resumed_gpu3/inference.jsonl"
OUT="$RUN_ROOT/eval_val_matched_diagnostic_from_resumed_official"

export PYTHONPATH=/home/weiminghong/humanslot:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

cd "$REPO"
mkdir -p "$OUT"
exec >> "$OUT/eval.log" 2>&1

echo "START $(date)"
echo "[matched] inference=$INFER"
echo "[matched] output=$OUT/metrics.json"
"$PY" -u human_lisa_qwen_v1/eval_matched_from_inference_v1.py   --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json   --image-root /share2/program/weiminghong/coco/val2017   --inference "$INFER"   --output "$OUT/metrics.json"   --wait   --poll-seconds 300   --iou-thr 0.5
echo "DONE $(date)"
