#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_scaledloss_v7_full_20260713_stage1full_final16029_dinov3_hqnet300_img640_e1_acc4_gpu1
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/train/final-step-16029
OUT=/share2/program/weiminghong/humanslot_outputs/stage2_scaledloss_v7_full_20260713_stage1full_final16029_dinov3_hqnet300_img640_e1_acc4_gpu1/eval_val_official_latest_gpu1
LOG="$RUN/eval_wait_latest_gpu1.log"
mkdir -p "$OUT"
{
  echo "wait_start_utc=$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
  echo "run=$RUN"
  echo "output=$OUT"
  echo "gpu=1"
} | tee -a "$LOG"
while true; do
  if pgrep -af "train_stage2_qwen_dino_hqnet_native.py" | grep -F "$RUN" >/dev/null 2>&1; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') training still running" | tee -a "$LOG"
    sleep 300
    continue
  fi
  if [[ -s "$RUN/latest.pt" ]]; then
    echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') found latest.pt, start eval" | tee -a "$LOG"
    break
  fi
  echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') training not running but latest.pt missing; wait" | tee -a "$LOG"
  sleep 300
done
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
"$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
  --profile full \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1/base_or_adapter" \
  --checkpoint "$RUN/latest.pt" \
  --output-dir "$OUT" \
  --image-size 640 \
  --num-query 300 \
  --batch-size 1 \
  --cuda-visible-devices 1 \
  --score-thr 0.3 \
  --official-score-thr 0.0 \
  --log-interval 50 \
  2>&1 | tee -a "$OUT/eval.log"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') eval done" | tee -a "$LOG"
