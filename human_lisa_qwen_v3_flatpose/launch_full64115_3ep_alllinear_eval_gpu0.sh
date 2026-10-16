#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/full64115_3ep_bs8ga2_alllinear_20260723
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-2}

if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 3
fi
mkdir -p "$(dirname "$RUN")"

COMMON_ARGS=(
  --output-dir "$RUN"
  --run-kind full
  --epochs 3
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
  --log-every 50
)

checkpoint_for_epoch() {
  local epoch=$1
  local matches=()
  shopt -s nullglob
  matches=("$RUN/epoch-${epoch}-step-"*)
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 || ! -d "${matches[0]}" ]]; then
    echo "Expected exactly one epoch-${epoch} checkpoint, found ${#matches[@]}" >&2
    exit 4
  fi
  printf '%s\n' "${matches[0]}"
}

evaluate_epoch() {
  local epoch=$1
  local checkpoint=$2
  "$PYTHON" -u human_lisa_qwen_v3_flatpose/eval_teacher_forced_v3.py \
    --checkpoint "$checkpoint" \
    --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
    --image-root /root/autodl-tmp/coco/val2017 \
    --output "$RUN/teacher_val_full_epoch${epoch}.json" \
    --max-samples 2693 --batch-size 8 \
    2>&1 | tee "$RUN.teacher_val_full_epoch${epoch}.log"
  "$PYTHON" -u human_lisa_qwen_v3_flatpose/eval_stage1_v3.py \
    --checkpoint "$checkpoint" \
    --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
    --image-root /root/autodl-tmp/coco/val2017 \
    --output-dir "$RUN/generated_val_full_epoch${epoch}_bs${EVAL_BATCH_SIZE}" \
    --batch-size "$EVAL_BATCH_SIZE" --max-new-tokens 5000 \
    2>&1 | tee "$RUN.generated_val_full_epoch${epoch}.log"
}

"$PYTHON" -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py \
  "${COMMON_ARGS[@]}" --stop-after-epoch 1 \
  2>&1 | tee "$RUN.train_epoch1.log"
CKPT1=$(checkpoint_for_epoch 1)
evaluate_epoch 1 "$CKPT1"

"$PYTHON" -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py \
  "${COMMON_ARGS[@]}" --resume-from "$CKPT1" --stop-after-epoch 2 \
  2>&1 | tee "$RUN.train_epoch2.log"
CKPT2=$(checkpoint_for_epoch 2)
evaluate_epoch 2 "$CKPT2"

"$PYTHON" -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py \
  "${COMMON_ARGS[@]}" --resume-from "$CKPT2" \
  2>&1 | tee "$RUN.train_epoch3.log"
CKPT3=$(checkpoint_for_epoch 3)
evaluate_epoch 3 "$CKPT3"

printf '%s\n' "Stage1 V3 full train and all three full-val evaluations completed."
