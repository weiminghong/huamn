#!/usr/bin/env bash
echo "[legacy] This script uses older stage1_sft_data JSONL/eval_lisa path; do not use for current official COCO-UniHuman v1 five-task Stage1. Use train_stage1_v1.py/eval_stage1_v1.py."
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
PY=/home/weiminghong/yes/envs/human_v2/bin/python
TRAIN_RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4
VAL_JSON=/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/val.jsonl
MODEL=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct
CKPT="$TRAIN_RUN/train/final"
OUT="$TRAIN_RUN/eval_val_full_final"
mkdir -p "$OUT"
{
  echo "started_wait_at=$(date -Is)"
  echo "stage=wait_for_stage1_final"
  echo "split=val"
  echo "val_json=${VAL_JSON}"
  echo "checkpoint=${CKPT}"
  echo "model=${MODEL}"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
} | tee "$OUT/eval_status.log"
# Wait up to 30 hours; training should finish much sooner.
for i in $(seq 1 10800); do
  if [ -s "$CKPT/token_heads.pt" ] && [ -s "$CKPT/base_or_adapter/adapter_model.safetensors" ] && [ -s "$CKPT/processor/tokenizer_config.json" ]; then
    if ! pgrep -f 'train_lisa_qwen_v1.py.*stage1_epoch1_20260709_qwen25_visual_lora_effbs4' >/dev/null; then
      break
    fi
  fi
  if [ $((i % 60)) -eq 0 ]; then
    echo "waiting_seconds=$((i*10)) time=$(date -Is)" | tee -a "$OUT/eval_status.log"
  fi
  sleep 10
done
if [ ! -s "$CKPT/token_heads.pt" ] || [ ! -s "$CKPT/base_or_adapter/adapter_model.safetensors" ] || [ ! -s "$CKPT/processor/tokenizer_config.json" ]; then
  echo "stage1_final_missing_after_wait time=$(date -Is)" | tee -a "$OUT/eval_status.log"
  exit 2
fi
if pgrep -f 'train_lisa_qwen_v1.py.*stage1_epoch1_20260709_qwen25_visual_lora_effbs4' >/dev/null; then
  echo "stage1_process_still_running_after_wait time=$(date -Is)" | tee -a "$OUT/eval_status.log"
  exit 3
fi
{
  echo "eval_started_at=$(date -Is)"
  echo "stage=eval_val_full_final"
  wc -l "$VAL_JSON"
  ls -lh "$CKPT/token_heads.pt" "$CKPT/base_or_adapter/adapter_model.safetensors" "$CKPT/processor/tokenizer_config.json"
} | tee -a "$OUT/eval_status.log"
"$PY" human_lisa_qwen_v1/eval_lisa_qwen_v1.py \
  --jsonl "$VAL_JSON" \
  --model "$MODEL" \
  --checkpoint "$CKPT" \
  --output-dir "$OUT" \
  --max-samples 999999 \
  --max-new-tokens 1024 \
  --bf16 \
  --cuda-visible-devices 2 \
  2>&1 | tee "$OUT/eval.log"
{
  echo "eval_finished_at=$(date -Is)"
  echo "stage=done_eval_val_full_final"
  if [ -s "$OUT/metrics.json" ]; then cat "$OUT/metrics.json"; fi
} | tee -a "$OUT/eval_status.log"
