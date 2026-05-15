#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-timm/vit_base_patch16_dinov3.lvd1689m}"
LOCAL_DIR="${LOCAL_DIR:-/share2/program/DINOV3/DINOV3-safetensor/dinov3-vitb16-pretrain-lvd1689m}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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
