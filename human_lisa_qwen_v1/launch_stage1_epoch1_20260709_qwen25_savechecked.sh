#!/usr/bin/env bash
echo "[legacy] This script uses older stage1_sft_data JSONL/eval_lisa path; do not use for current official COCO-UniHuman v1 five-task Stage1. Use train_stage1_v1.py/eval_stage1_v1.py."
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
OUT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_savechecked
PY=/home/weiminghong/yes/envs/human_v2/bin/python
TRAIN_JSON=/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/stage1_sft_data/train.jsonl
MODEL=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct
mkdir -p "$OUT"
{
  echo "started_at=$(date -Is)"
  echo "stage=train_epoch1"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "model=${MODEL}"
  echo "train_json=${TRAIN_JSON}"
  echo "output_dir=${OUT}/train"
  echo "epochs=1"
  echo "max_steps=0"
  echo "batch_size=1"
  echo "max_length=1024"
  echo "lora=true"
  echo "lora_r=8"
  echo "lora_alpha=16"
  echo "lr=2e-5"
  echo "head_lr=1e-4"
  echo "bf16=true"
  echo "save_every=8000"
} | tee "$OUT/status.log"
"$PY" human_lisa_qwen_v1/train_lisa_qwen_v1.py \
  --train-jsonl "$TRAIN_JSON" \
  --model "$MODEL" \
  --output-dir "$OUT/train" \
  --epochs 1 \
  --max-steps 0 \
  --batch-size 1 \
  --max-length 1024 \
  --lora --lora-r 8 --lora-alpha 16 \
  --lr 2e-5 \
  --head-lr 1e-4 \
  --log-every 100 \
  --save-every 8000 \
  --bf16 \
  --cuda-visible-devices 2 \
  2>&1 | tee "$OUT/train.log"
FINAL="$OUT/train/final"
for f in "$FINAL/token_heads.pt" "$FINAL/base_or_adapter/adapter_model.safetensors" "$FINAL/processor/tokenizer_config.json"; do
  if [ ! -s "$f" ]; then
    echo "missing_final_file=$f" | tee -a "$OUT/status.log"
    exit 2
  fi
done
{
  echo "finished_at=$(date -Is)"
  echo "stage=done_train_final_verified"
  ls -lh "$FINAL/token_heads.pt" "$FINAL/base_or_adapter/adapter_model.safetensors" "$FINAL/processor/tokenizer_config.json"
} | tee -a "$OUT/status.log"
