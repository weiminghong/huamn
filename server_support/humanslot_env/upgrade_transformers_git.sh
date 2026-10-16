#!/usr/bin/env bash
set -euo pipefail
LOG=/root/autodl-tmp/humanslot_env/upgrade_transformers_git.log
{
  echo "[start] $(date -Is)"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate human4b
  python -m pip install -U 'git+https://github.com/huggingface/transformers.git'
  python - <<'PY'
import transformers, torch
print('transformers', transformers.__version__)
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    print('has_qwen3_5', 'qwen3_5' in CONFIG_MAPPING)
except Exception as e:
    print('config_mapping_check_error', repr(e))
print('torch', torch.__version__, torch.cuda.is_available())
PY
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG"
