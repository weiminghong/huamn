#!/usr/bin/env bash
set -euo pipefail
OLD='weiminghong@221.204.195.130'
SSH=(ssh -i /root/.ssh/humanslot_transfer_ed25519 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/root/.ssh/known_hosts -p 230)
LOG=/root/autodl-tmp/humanslot_transfer/referenced_train_transfer.log
{
  echo "[start] $(date -Is)"
  rsync -ah --info=progress2 --partial --files-from=/root/autodl-tmp/humanslot_transfer/train2017_unihuman_files.txt -e "${SSH[*]}" \
    "$OLD:/share2/program/weiminghong/coco/train2017/" /root/autodl-tmp/coco/train2017/
  echo "[verify]"
  /root/miniconda3/envs/human4b/bin/python - <<'PY'
from pathlib import Path
root=Path('/root/autodl-tmp/coco/train2017')
files=[line.strip() for line in Path('/root/autodl-tmp/humanslot_transfer/train2017_unihuman_files.txt').read_text().splitlines() if line.strip()]
missing=[f for f in files if not (root/f).is_file()]
print({'required': len(files), 'present': len(files)-len(missing), 'missing': len(missing), 'first_missing': missing[:5]})
if missing:
    raise SystemExit(7)
PY
  du -sh /root/autodl-tmp/coco/train2017 /root/autodl-tmp/coco/val2017 /root/autodl-tmp/coco/unihuman_label
  df -h /root/autodl-tmp
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG"
