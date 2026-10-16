#!/usr/bin/env bash
set -uo pipefail
RUN="/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/smoke_real_qwen35_vision_tuned_5step_20260805"
mkdir -p "$RUN"
cd /root/autodl-tmp/humanslot || exit 90
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
printf 'STARTED %s
' "$(date -Is)" > "$RUN/status.log"
set +e
/root/miniconda3/envs/human4b/bin/python -u -m human_lisa_qwen_v4.train_stage1_v4 \
  --model /root/autodl-tmp/model/Qwen3.5-4B \
  --output-dir "$RUN/run" \
  --profile vision_tuned \
  --max-samples 96 \
  --epochs 1 \
  --max-optimizer-steps 5 \
  --batch-size 4 \
  --grad-accum-steps 4 \
  --num-workers 2 \
  --max-length 5000 \
  --decoder-warmup-steps 2 \
  --log-every 1 \
  --save-every 0 \
  --seed 42 \
  --bf16 2>&1 | tee "$RUN/train.log"
rc=${PIPESTATUS[0]}
set -e
printf 'FINISHED %s exit_code=%s
' "$(date -Is)" "$rc" >> "$RUN/status.log"
exit "$rc"
