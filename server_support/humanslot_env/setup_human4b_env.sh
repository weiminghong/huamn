#!/usr/bin/env bash
set -euo pipefail
LOG=/root/autodl-tmp/humanslot_env/setup.log
{
  echo "[start] $(date -Is)"
  source /root/miniconda3/etc/profile.d/conda.sh
  if ! conda env list | awk '{print $1}' | grep -qx human4b; then
    conda create -y -n human4b python=3.10
  fi
  conda activate human4b
  python -m pip install -U pip setuptools wheel
  # Install core runtime. Torch wheel is selected by pip; driver supports CUDA 13.1 runtime compatibility.
  python -m pip install 'torch>=2.3,<2.7' 'torchvision>=0.18,<0.22'
  python -m pip install -r /root/autodl-tmp/humanslot/requirements.txt
  python -m pip install pycocotools jsonschema jsonlines
  python - <<'PY'
import torch, transformers, peft
print('python-ok')
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())
print('transformers', transformers.__version__)
print('peft', peft.__version__)
PY
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG"
