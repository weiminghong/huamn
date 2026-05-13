#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-timm/vit_base_patch16_dinov3.lvd1689m}"
LOCAL_DIR="${LOCAL_DIR:-/home/priv/wmh_project/models/dinov3/vit_base_patch16_dinov3.lvd1689m}"
PYTHON_BIN="${PYTHON_BIN:-/home/priv/miniconda3/envs/human_v2/bin/python}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "${LOCAL_DIR}"
"${PYTHON_BIN}" - <<PY
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="${MODEL_ID}",
    local_dir="${LOCAL_DIR}",
    local_dir_use_symlinks=False,
    max_workers=1,
    etag_timeout=30,
)
print("Downloaded ${MODEL_ID} to ${LOCAL_DIR}")
PY
