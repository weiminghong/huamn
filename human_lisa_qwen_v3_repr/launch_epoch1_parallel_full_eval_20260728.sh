#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/humanslot
PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/full64115_2ep_bs8ga2_monitor_20260728
CKPT="$RUN/epoch-1-step-4008"
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
IMAGES=/root/autodl-tmp/coco/val2017
QUICK="$RUN/eval_epoch1_generated_val32_tok5000_bs1"
TF="$RUN/eval_epoch1_tf_full_val2693_bs1"
GEN="$RUN/eval_epoch1_generated_full_val2693_bs1"
STATUS="$RUN/eval_epoch1_parallel_full.status.log"

exec 9>"$RUN/.eval_epoch1_parallel_full.lock"
if ! flock -n 9; then
  echo "[$(date '+%F %T %Z')] duplicate launch refused: lock held" >> "$STATUS"
  exit 0
fi
trap 'rc=$?; printf "[%s] epoch1 parallel evaluation pipeline exit rc=%s\n" "$(date "+%F %T %Z")" "$rc" | tee -a "$STATUS"' EXIT

for required in "$CKPT/manifest.json" "$CKPT/base_or_adapter/adapter_model.safetensors" "$CKPT/stage1_v3_task_modules.pt"; do
  [[ -f "$required" ]] || { echo "missing checkpoint artifact: $required"; exit 4; }
done
cd "$ROOT"
printf '%s  %s\n' \
  5cf973970a43a9759ccbbe71dbd97191f324446d9dc236a5e1cad97945a29c09 human_lisa_qwen_v3_repr/eval_stage1_v3.py \
  b4938ed5de329e306fc85e4b10fde29eba93117766681d3f2db5c73b8098a02c human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
  | sha256sum -c -
if pgrep -af 'eval_teacher_forced_v3.py|eval_stage1_v3.py' | grep -v "$$" >/dev/null; then
  echo "[$(date '+%F %T %Z')] duplicate evaluator process refused" | tee -a "$STATUS"
  exit 5
fi
read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ' | tr ',' ' ')
FREE=$((TOTAL - USED))
(( FREE >= 24576 )) || { echo "[$(date '+%F %T %Z')] insufficient GPU headroom: ${FREE} MiB" | tee -a "$STATUS"; exit 6; }

mkdir -p "$QUICK" "$TF" "$GEN"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '[%s] quick generated val32 start; bs=1 max_new_tokens=5000 free_gpu_mib=%s\n' "$(date '+%F %T %Z')" "$FREE" | tee -a "$STATUS"
nice -n 10 "$PYTHON" -u human_lisa_qwen_v3_repr/eval_stage1_v3.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output-dir "$QUICK" --max-samples 32 --batch-size 1 --max-new-tokens 5000 \
  2>&1 | tee "$QUICK/run.log"
printf '[%s] quick generated val32 complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"

printf '[%s] teacher-forced full val2693 start; bs=1\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
nice -n 10 "$PYTHON" -u human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output "$TF/metrics.json" --max-samples 2693 --batch-size 1 \
  2>&1 | tee "$TF/run.log"
printf '[%s] teacher-forced full val2693 complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"

printf '[%s] generated full val2693 start; bs=1 max_new_tokens=5000\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
nice -n 10 "$PYTHON" -u human_lisa_qwen_v3_repr/eval_stage1_v3.py \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output-dir "$GEN" --batch-size 1 --max-new-tokens 5000 \
  2>&1 | tee "$GEN/run.log"
printf '[%s] generated full val2693 complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
