#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/humanslot
PYTHON=/root/miniconda3/envs/human4b/bin/python
BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/full64115_2ep_bs8ga2_20260805
RUN="$BASE/run"
CKPT="$RUN/epoch-1-step-4008"
ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
IMAGES=/root/autodl-tmp/coco/val2017
TF="$BASE/eval_epoch1_tf_full_val2693_bs1_20260807"
GEN="$BASE/eval_epoch1_generated_full_val2693_bs1_tok5000_20260807"
STATUS="$BASE/eval_epoch1_full_20260807.status.log"
LOCK="$BASE/.eval_epoch1_full_20260807.lock"

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "[$(date '+%F %T %Z')] duplicate launch refused: lock held" | tee -a "$STATUS"
  exit 0
fi
trap 'rc=$?; printf "[%s] v4 epoch1 full eval exit rc=%s\n" "$(date "+%F %T %Z")" "$rc" | tee -a "$STATUS"' EXIT

for required in "$CKPT/_SUCCESS" "$CKPT/base_or_adapter/adapter_model.safetensors" "$CKPT/stage1_v4_tasks.pt" "$CKPT/visual_merger.pt"; do
  [[ -f "$required" ]] || { echo "missing checkpoint artifact: $required" | tee -a "$STATUS"; exit 4; }
done
[[ -x "$PYTHON" ]] || { echo "missing python env: $PYTHON" | tee -a "$STATUS"; exit 4; }
[[ -f "$ANN" ]] || { echo "missing annotation: $ANN" | tee -a "$STATUS"; exit 4; }
[[ -d "$IMAGES" ]] || { echo "missing image root: $IMAGES" | tee -a "$STATUS"; exit 4; }
cd "$ROOT"

if pgrep -af '/root/miniconda3/envs/human4b/bin/python .*human_lisa_qwen_v4\.eval_(teacher_forced_v4|stage1_v4)' >/dev/null; then
  echo "[$(date '+%F %T %Z')] duplicate evaluator process refused" | tee -a "$STATUS"
  pgrep -af '/root/miniconda3/envs/human4b/bin/python .*human_lisa_qwen_v4\.eval_(teacher_forced_v4|stage1_v4)' | tee -a "$STATUS"
  exit 5
fi
read -r USED TOTAL < <(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -n1 | tr -d ' ' | tr ',' ' ')
FREE=$((TOTAL - USED))
if (( FREE < 24576 )); then
  echo "[$(date '+%F %T %Z')] insufficient GPU headroom: ${FREE} MiB" | tee -a "$STATUS"
  exit 6
fi
mkdir -p "$TF" "$GEN"
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$ROOT"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

printf '[%s] teacher-forced full nUIHuman val start; images=2693 bs=1 free_gpu_mib=%s\n' "$(date '+%F %T %Z')" "$FREE" | tee -a "$STATUS"
nice -n 10 "$PYTHON" -u -m human_lisa_qwen_v4.eval_teacher_forced_v4 \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output-dir "$TF" --batch-size 1 --num-workers 2 --max-length 5000 \
  2>&1 | tee "$TF/run.log"
printf '[%s] teacher-forced full nUIHuman val complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"

printf '[%s] generated full nUIHuman val start; images=2693 bs=1 max_new_tokens=5000\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
nice -n 10 "$PYTHON" -u -m human_lisa_qwen_v4.eval_stage1_v4 \
  --checkpoint "$CKPT" --annotation "$ANN" --image-root "$IMAGES" \
  --output-dir "$GEN" --batch-size 1 --max-new-tokens 5000 \
  2>&1 | tee "$GEN/run.log"
printf '[%s] generated full nUIHuman val complete\n' "$(date '+%F %T %Z')" | tee -a "$STATUS"
