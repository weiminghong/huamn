#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/gates/resume_epoch_boundary_alllinear_20260723
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 3
fi
COMMON=(
  --output-dir "$OUT" --run-kind smoke --max-samples 4 --epochs 3
  --batch-size 4 --grad-accum-steps 1 --num-workers 4 --horizontal-flip-prob 0
  --max-people 25 --person-query-layers 3 --pose-decoder-layers 2
  --lora-target-mode all-linear --scheduler-type cosine --warmup-ratio 0.05
  --visual-lr 1e-5 --language-lr 1e-5 --token-lr 2e-5 --head-lr 2e-4
  --weight-decay 0.01 --grad-clip 1
  --query-objectness-weight 1 --query-bbox-l1-weight 5 --query-giou-weight 2
  --query-aux-weight 0.5 --pose-l1-weight 10 --pose-oks-weight 4
  --pose-visibility-weight 1 --pose-aux-weight 0.5
  --mask-bce-weight 2 --mask-dice-weight 5 --log-every 1
)
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py   "${COMMON[@]}" --stop-after-epoch 1 2>&1 | tee "$OUT.pause_after_epoch1.log"
CKPT="$OUT/epoch-1-step-1"
test -f "$CKPT/training_state.pt"
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py   "${COMMON[@]}" --resume-from "$CKPT" 2>&1 | tee "$OUT.resume_epoch2_to3.log"
test -d "$OUT/epoch-2-step-2"
test -d "$OUT/epoch-3-step-3"
test -d "$OUT/final-step-3"
