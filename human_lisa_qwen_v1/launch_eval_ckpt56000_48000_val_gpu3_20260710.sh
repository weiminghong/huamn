#!/usr/bin/env bash
echo "[legacy] This script uses older stage1_sft_data JSONL/eval_lisa path; do not use for current official COCO-UniHuman v1 five-task Stage1. Use train_stage1_v1.py/eval_stage1_v1.py."
set -euo pipefail
cd /home/weiminghong/humanslot
GPU="${GPU:-3}"
BS="${BS:-1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1024}"
MAX_SAMPLES="${MAX_SAMPLES:-999999}"
CKPTS="${CKPTS:-checkpoint-56000 checkpoint-48000}"
OUT_TAG="${OUT_TAG:-gpu${GPU}_bs${BS}_tok${MAX_NEW_TOKENS}}"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONUNBUFFERED=1
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
PY=/home/weiminghong/yes/envs/human_v2/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4
VAL_JSON=/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/val.jsonl
MODEL=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct
for CK in $CKPTS; do
  CKPT="$RUN/train/$CK"
  OUT="$RUN/eval_val_full_${CK}_${OUT_TAG}"
  mkdir -p "$OUT"
  {
    echo "eval_started_at=$(date -Is)"
    echo "split=val"
    echo "checkpoint=${CKPT}"
    echo "output_dir=${OUT}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "batch_size=${BS}"
    echo "max_new_tokens=${MAX_NEW_TOKENS}"
    echo "max_samples=${MAX_SAMPLES}"
    echo "out_tag=${OUT_TAG}"
    wc -l "$VAL_JSON"
    ls -lh "$CKPT/token_heads.pt" "$CKPT/base_or_adapter/adapter_model.safetensors" "$CKPT/processor/tokenizer_config.json"
  } | tee "$OUT/eval_status.log"
  "$PY" human_lisa_qwen_v1/eval_lisa_qwen_v1.py \
    --jsonl "$VAL_JSON" \
    --model "$MODEL" \
    --checkpoint "$CKPT" \
    --output-dir "$OUT" \
    --max-samples "$MAX_SAMPLES" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --batch-size "$BS" \
    --bf16 \
    --cuda-visible-devices "$GPU" \
    2>&1 | tee "$OUT/eval.log"
  {
    echo "eval_finished_at=$(date -Is)"
    echo "stage=done_eval_val_full_${CK}"
    if [ -s "$OUT/metrics.json" ]; then cat "$OUT/metrics.json"; fi
  } | tee -a "$OUT/eval_status.log"
done
