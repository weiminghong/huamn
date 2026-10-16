#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/full64115_2ep_bs8ga2_monitor_20260728
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}
STATUS=${RUN}.status

if [[ -e "$RUN" || -e "${RUN}.train.log" ]]; then
  printf 'Refusing to overwrite existing run or log: %s\n' "$RUN" >&2
  exit 3
fi
mkdir -p "$(dirname "$RUN")"
printf 'phase=launching time=%s\n' "$(date -Is)" > "$STATUS"
sha256sum "$0" > "${RUN}.launcher.sha256"

COMMON_ARGS=(
  --output-dir "$RUN"
  --run-kind full
  --epochs 2
  --batch-size 8
  --grad-accum-steps 2
  --max-length 5000
  --num-workers 4
  --horizontal-flip-prob 0.5
  --max-people 25
  --person-query-layers 3
  --pose-decoder-layers 2
  --lora-target-mode all-linear
  --scheduler-type cosine
  --warmup-ratio 0.05
  --visual-lr 1e-5
  --language-lr 1e-5
  --token-lr 2e-5
  --head-lr 2e-4
  --weight-decay 0.01
  --grad-clip 1
  --query-objectness-weight 1
  --query-bbox-l1-weight 5
  --query-giou-weight 2
  --query-aux-weight 0.5
  --pose-l1-weight 10
  --pose-oks-weight 4
  --pose-visibility-weight 1
  --pose-aux-weight 0.5
  --mask-bce-weight 2
  --mask-dice-weight 5
  --log-every 25
  --monitor-every 250
  --monitor-samples 2
  --monitor-seed 20260728
)

checkpoint_for_epoch() {
  local epoch=$1
  local matches=()
  shopt -s nullglob
  matches=("$RUN/epoch-${epoch}-step-"*)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    printf 'Expected exactly one epoch-%s checkpoint, found %s\n' "$epoch" "${#matches[@]}" >&2
    exit 4
  fi
  printf '%s\n' "${matches[0]}"
}

evaluate_epoch() {
  local epoch=$1
  local checkpoint=$2
  printf 'phase=teacher_eval epoch=%s checkpoint=%s time=%s\n' "$epoch" "$checkpoint" "$(date -Is)" > "$STATUS"
  "$PYTHON" -u human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
    --checkpoint "$checkpoint" \
    --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
    --image-root /root/autodl-tmp/coco/val2017 \
    --output "$RUN/teacher_val_full_epoch${epoch}.json" \
    --max-samples 2693 --batch-size 8 \
    2>&1 | tee "${RUN}.teacher_val_full_epoch${epoch}.log"
  printf 'phase=generated_eval epoch=%s checkpoint=%s time=%s\n' "$epoch" "$checkpoint" "$(date -Is)" > "$STATUS"
  "$PYTHON" -u human_lisa_qwen_v3_repr/eval_stage1_v3.py \
    --checkpoint "$checkpoint" \
    --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
    --image-root /root/autodl-tmp/coco/val2017 \
    --output-dir "$RUN/generated_val_full_epoch${epoch}_bs${EVAL_BATCH_SIZE}" \
    --batch-size "$EVAL_BATCH_SIZE" --max-new-tokens 5000 \
    2>&1 | tee "${RUN}.generated_val_full_epoch${epoch}.log"
}

printf 'phase=training time=%s\n' "$(date -Is)" > "$STATUS"
"$PYTHON" -u human_lisa_qwen_v3_repr/train_stage1_v3.py \
  "${COMMON_ARGS[@]}" \
  2>&1 | tee "${RUN}.train.log"

cp -- "$0" "$RUN/launch.sh"
cp -- "${RUN}.launcher.sha256" "$RUN/launcher.sha256"
CKPT1=$(checkpoint_for_epoch 1)
CKPT2=$(checkpoint_for_epoch 2)
evaluate_epoch 1 "$CKPT1"
evaluate_epoch 2 "$CKPT2"
printf 'phase=completed time=%s\n' "$(date -Is)" > "$STATUS"
printf 'Stage1 V3 representation training and both full-val evaluations completed.\n'
