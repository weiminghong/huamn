#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYTHON=/root/miniconda3/envs/human4b/bin/python
RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/gates/overfit8_boxfree_multiscale_500steps_20260725
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite existing run: $RUN" >&2
  exit 3
fi
mkdir -p "$(dirname "$RUN")"

"$PYTHON" -u human_lisa_qwen_v3_repr/train_stage1_v3.py \
  --output-dir "$RUN" \
  --run-kind overfit \
  --max-samples 8 \
  --epochs 500 \
  --max-optimizer-steps 500 \
  --batch-size 8 \
  --grad-accum-steps 1 \
  --max-length 5000 \
  --num-workers 4 \
  --horizontal-flip-prob 0 \
  --max-people 25 \
  --person-query-layers 3 \
  --pose-decoder-layers 2 \
  --lora-target-mode all-linear \
  --scheduler-type constant \
  --warmup-ratio 0.02 \
  --visual-lr 1e-5 \
  --language-lr 1e-5 \
  --token-lr 2e-5 \
  --head-lr 1e-3 \
  --weight-decay 0.01 \
  --grad-clip 5 \
  --query-objectness-weight 1 \
  --query-bbox-l1-weight 5 \
  --query-giou-weight 2 \
  --query-aux-weight 0.5 \
  --pose-l1-weight 10 \
  --pose-oks-weight 4 \
  --pose-visibility-weight 1 \
  --pose-aux-weight 0.5 \
  --mask-bce-weight 2 \
  --mask-dice-weight 5 \
  --log-every 10 \
  --no-save-epoch-checkpoints \
  2>&1 | tee "$RUN.train.log"

"$PYTHON" -u human_lisa_qwen_v3_repr/eval_teacher_forced_v3.py \
