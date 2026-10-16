#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/gates/stress_batch8_alllinear_20260723.json
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 3
fi
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_boxpose/stress_batch_v3.py   --output "$OUT"   --indices 23708 31301 62691 62661 62059 61823 61665 59900   2>&1 | tee /root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_boxpose/gates/stress_batch8_alllinear_20260723.log
