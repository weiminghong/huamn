#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/humanslot
PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/full64115_2ep_bs8ga2_monitor_20260728
CKPT="$RUN/epoch-1-step-4008"
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
IMAGES=/root/autodl-tmp/coco/val2017
TF_DIR="$RUN/eval_epoch1_tf_val64"
GEN_DIR="$RUN/eval_epoch1_generated_val32"
STATUS="$RUN/eval_epoch1_small.status.log"

exec 9>"$RUN/.eval_epoch1_small.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T %Z')] another epoch-1 small evaluator holds the lock" >> "$STATUS"
  exit 0
fi

if [[ ! -f "$CKPT/manifest.json" || ! -f "$CKPT/base_or_adapter/adapter_model.safetensors" || ! -f "$CKPT/stage1_v3_task_modules.pt" ]]; then
  echo "[$(date '+%F %T %Z')] checkpoint is incomplete: $CKPT" | tee -a "$STATUS"
  exit 4
fi
if pgrep -af 'eval_teacher_forced_v3.py|eval_stage1_v3.py' | grep -v "$$" >/dev/null; then
  echo "[$(date '+%F %T %Z')] refusing duplicate evaluator" | tee -a "$STATUS"
  exit 5
fi

read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ' | tr ',' ' ')
FREE=$((TOTAL - USED))
if (( FREE < 24576 )); then
  echo "[$(date '+%F %T %Z')] insufficient GPU headroom: ${FREE} MiB" | tee -a "$STATUS"
  exit 6
fi

mkdir -p "$TF_DIR" "$GEN_DIR"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '[%s] teacher-forced val64 start; free_gpu_mib=%s\n' "$(date '+%F %T %Z')" "$FREE" | tee -a "$STATUS"
"$PYTHON" -u human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
  --checkpoint "$CKPT" \
  --annotation "$ANN" \
  --image-root "$IMAGES" \
  --output "$TF_DIR/metrics.json" \
  --max-samples 64 \
  --batch-size 1 \
  2>&1 | tee "$TF_DIR/run.log"
printf '[%s] teacher-forced val64 complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"

printf '[%s] generated val32 start\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
"$PYTHON" -u human_lisa_qwen_v3_repr/eval_stage1_v3.py \
  --checkpoint "$CKPT" \
  --annotation "$ANN" \
  --image-root "$IMAGES" \
  --output-dir "$GEN_DIR" \
  --max-samples 32 \
  --batch-size 1 \
  --max-new-tokens 2048 \
  2>&1 | tee "$GEN_DIR/run.log"
printf '[%s] generated val32 complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
