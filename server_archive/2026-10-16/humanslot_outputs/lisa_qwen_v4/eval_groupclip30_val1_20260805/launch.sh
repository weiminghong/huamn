#!/usr/bin/env bash
set -uo pipefail
RUN="/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/eval_groupclip30_val1_20260805"
mkdir -p "$RUN"
cd /root/autodl-tmp/humanslot || exit 90
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
printf 'STARTED %s\n' "$(date -Is)" > "$RUN/status.log"
set +e
timeout 900 /root/miniconda3/envs/human4b/bin/python -u -m human_lisa_qwen_v4.eval_stage1_v4 \
  --checkpoint /root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/smoke_group_clip_dynamic_qwen35_30step_20260805/run/epoch-1-step-30 \
  --output-dir "$RUN/output" \
  --max-samples 1 \
  --batch-size 1 \
  --max-new-tokens 5000 \
  --bf16 2>&1 | tee "$RUN/eval.log"
rc=${PIPESTATUS[0]}
printf 'FINISHED %s exit_code=%s\n' "$(date -Is)" "$rc" >> "$RUN/status.log"
exit "$rc"
