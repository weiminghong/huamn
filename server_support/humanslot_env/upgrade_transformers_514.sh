#!/usr/bin/env bash
set -euo pipefail
LOG=/root/autodl-tmp/humanslot_env/upgrade_transformers_514.log
{
  echo "[start] $(date -Is)"
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate human4b
  python -m pip install -U 'transformers==5.14.1'
  python - <<'PY'
import transformers, pkgutil
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
print('transformers', transformers.__version__)
print('has_qwen3_5', 'qwen3_5' in CONFIG_MAPPING)
print([m.name for m in pkgutil.iter_modules(transformers.models.__path__) if 'qwen3' in m.name])
PY
  echo "[done] $(date -Is)"
} 2>&1 | tee "$LOG"
