#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260710_stage1final_qwen25_visual_lora_bs4_e1
OUT="$RUN/eval_val_native_latest_gpu2"
CKPT="$RUN/latest.pt"
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4/train/final/base_or_adapter
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
{
  echo "started_at=$(date -Is)"
  echo "stage=eval_stage2_latest_val_native"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "checkpoint=${CKPT}"
  echo "qwen_adapter=${STAGE1}"
  echo "output_dir=${OUT}"
  echo "score_thr=0.3"
  echo "protocol=stage2_native_224_single_scale_scorethr_diagnostic"
} | tee "$OUT/launch_status.log"
"$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
  --checkpoint "$CKPT" \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1" \
  --output-dir "$OUT" \
  --cuda-visible-devices 2 \
  --score-thr 0.3 \
  --log-interval 20 \
  2>&1 | tee -a "$OUT/eval.log"
{
  echo "finished_at=$(date -Is)"
  echo "stage=done_eval_stage2_latest_val_native"
  ls -lh "$OUT/metrics.json" "$OUT/predictions.jsonl" 2>/dev/null || true
} | tee -a "$OUT/launch_status.log"
