#!/usr/bin/env bash
set -euo pipefail
BASE=/root/autodl-tmp/humanslot_transfer
LOG="$BASE/transfer.log"
KEY=/root/.ssh/humanslot_transfer_ed25519
OLD='weiminghong@221.204.195.130'
SSH="ssh -i $KEY -p 230 -o BatchMode=yes -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ServerAliveCountMax=6"
mkdir -p /root/autodl-tmp/humanslot /root/autodl-tmp/model /root/autodl-tmp/humanslot_outputs/lisa_qwen_v1
{
  echo "[start] $(date -Is)"
  echo "[disk-before]"; df -h /root/autodl-tmp / || true
  echo "[1/5] rsync repo scripts"
  rsync -ah --info=progress2 --partial --delete \
    --exclude 'outputs/' --exclude '**/__pycache__/' --exclude '.git/' \
    -e "$SSH" "$OLD:/home/weiminghong/humanslot/" /root/autodl-tmp/humanslot/
  echo "[2/5] rsync Qwen3.5-4B"
  rsync -ah --info=progress2 --partial \
    -e "$SSH" "$OLD:/share2/program/weiminghong/model/Qwen3.5-4B/" /root/autodl-tmp/model/Qwen3.5-4B/
  echo "[3/5] rsync current Stage1 base model Qwen2.5-VL-3B-Instruct"
  rsync -ah --info=progress2 --partial \
    -e "$SSH" "$OLD:/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct/" /root/autodl-tmp/model/Qwen2.5-VL-3B-Instruct/
  echo "[4/5] rsync latest Stage1 checkpoint-16000"
  RUN=stage1_v1_stable5task_full_3epoch_20260716_044540_gpu2
  mkdir -p "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN/train"
  rsync -ah --info=progress2 --partial \
    -e "$SSH" "$OLD:/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/$RUN/train/checkpoint-16000/" \
    "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN/train/checkpoint-16000/"
  echo "[5/5] rsync compact run metadata and eval metrics"
  rsync -ah --info=progress2 --partial \
    -e "$SSH" \
    "$OLD:/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/$RUN/status.log" \
    "$OLD:/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/$RUN/train.log" \
    "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN/" || true
  mkdir -p "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN/eval_val_full_checkpoint-10000_gpu3"
  rsync -ah --info=progress2 --partial \
    -e "$SSH" \
    "$OLD:/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/$RUN/eval_val_full_checkpoint-10000_gpu3/metrics.json" \
    "$OLD:/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/$RUN/eval_val_full_checkpoint-10000_gpu3/predictions.coco_multitask.json" \
    "/root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN/eval_val_full_checkpoint-10000_gpu3/" || true
  echo "[sizes]"; du -sh /root/autodl-tmp/humanslot /root/autodl-tmp/model/* /root/autodl-tmp/humanslot_outputs/lisa_qwen_v1/$RUN 2>/dev/null || true
  echo "[disk-after]"; df -h /root/autodl-tmp / || true
  echo "[done] $(date -Is)"
} 2>&1 | tee -a "$LOG"
