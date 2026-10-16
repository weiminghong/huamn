#!/usr/bin/env bash
set -euo pipefail
OLD='weiminghong@221.204.195.130'
SSH=(ssh -i /root/.ssh/humanslot_transfer_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts -p 230)
RSYNC=(rsync -ah --info=progress2 --partial -e "${SSH[*]}")
LOG=/root/autodl-tmp/humanslot_transfer/data_transfer.log
{
  echo "[start] $(date -Is)"
  echo "[1/4] labels"
  "${RSYNC[@]}" "$OLD:/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json" /root/autodl-tmp/coco/unihuman_label/
  "${RSYNC[@]}" "$OLD:/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json" /root/autodl-tmp/coco/unihuman_label/
  echo "[2/4] val2017"
  mkdir -p /root/autodl-tmp/coco/val2017
  "${RSYNC[@]}" "$OLD:/share2/program/weiminghong/coco/val2017/" /root/autodl-tmp/coco/val2017/
  echo "[3/4] train2017"
  mkdir -p /root/autodl-tmp/coco/train2017
  "${RSYNC[@]}" "$OLD:/share2/program/weiminghong/coco/train2017/" /root/autodl-tmp/coco/train2017/
  echo "[4/4] sizes"
  du -sh /root/autodl-tmp/coco/unihuman_label /root/autodl-tmp/coco/val2017 /root/autodl-tmp/coco/train2017
  df -h /root/autodl-tmp
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG"
