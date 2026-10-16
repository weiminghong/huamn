#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export PYTHONPATH=/root/autodl-tmp/humanslot
PY=/root/miniconda3/envs/human4b/bin/python
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
INF=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/stage1_v1_qwen35_4b_5task_3epoch_epochval_20260718_223143_gpu0/eval_val_full_epoch1_final-step-5343/inference.jsonl
OUTDIR=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/diagnostics_epoch1_count_score_20260722
mkdir -p "$OUTDIR"
"$PY" -u human_lisa_qwen_v1/diagnose_stage1_posthoc_v1.py \
  --annotation "$ANN" \
  --inference "$INF" \
  --output "$OUTDIR/metrics.json" \
  2>&1 | tee "$OUTDIR/run.log"
