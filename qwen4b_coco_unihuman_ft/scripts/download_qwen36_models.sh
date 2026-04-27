#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT=${MODEL_ROOT:-/home/priv/project/model}
HF_ENDPOINT=${HF_ENDPOINT:-https://huggingface.co}

export HF_ENDPOINT

mkdir -p "${MODEL_ROOT}"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "huggingface-cli not found. Install it first:"
  echo "  pip install -U huggingface_hub"
  exit 1
fi

download_model() {
  local repo_id="$1"
  local local_name="$2"
  local local_dir="${MODEL_ROOT}/${local_name}"

  echo "Downloading ${repo_id}"
  echo "Target: ${local_dir}"

  huggingface-cli download "${repo_id}" \
    --local-dir "${local_dir}" \
    --local-dir-use-symlinks False \
    --resume-download
}

download_model "Qwen/Qwen3.6-35B-A3B" "Qwen3.6-35B-A3B"
download_model "Qwen/Qwen3.6-27B" "Qwen3.6-27B"

echo "All downloads finished."
echo "Model root: ${MODEL_ROOT}"
