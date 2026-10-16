#!/usr/bin/env bash
set -euo pipefail

REPO=/home/weiminghong/humanslot
PYTHON=/home/weiminghong/yes/envs/human_v2/bin/python
OUT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit32_v7_refpoint_joint_200steps_gpu3

export CUDA_VISIBLE_DEVICES=3
export HF_HOME=/share2/program/weiminghong/hf_cache
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers

if [[ -e "$OUT" ]]; then
  echo "refusing to overwrite existing output: $OUT" >&2
  exit 2
fi
mkdir -p "$OUT"
cd "$REPO"

"$PYTHON" -u human_lisa_qwen_v1/train_stage1_v1.py \
  --output-dir "$OUT/train" \
  --max-samples 32 \
  --epochs 25 \
  --max-optimizer-steps 200 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 5000 \
  --missing-attribute-loss-weight 0 \
  --ce-loss-weight 1 \
  --box-l1-loss-weight 5 \
  --box-giou-loss-weight 2 \
  --box-quality-loss-weight 1 \
  --pose-l1-loss-weight 5 \
  --pose-oks-loss-weight 1.5 \
  --pose-heatmap-loss-weight 0 \
  --pose-visibility-loss-weight 1 \
  --mask-bce-loss-weight 8 \
  --mask-dice-loss-weight 5 \
  --trainable-families visual language box pose mask \
  --num-workers 2 \
  --horizontal-flip-prob 0 \
  --base-lr 2e-5 \
  --visual-lr 5e-5 \
  --language-lr 2e-4 \
  --head-lr 1e-4 \
  --box-lr 5e-4 \
  --pose-lr 5e-4 \
  --mask-lr 2e-4 \
  --weight-decay 0 \
  --warmup-ratio 0.03 \
  --scheduler-type constant \
  --grad-clip 5 \
  --task-backbone-grad-scale 0.1 \
  --no-pose-box-jitter \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0 \
  --decoder-dim 256 \
  --decoder-heads 8 \
  --bf16 \
  --seed 42 \
  --log-every 10 \
  --save-every 200 \
  --probe-every 100 \
  --probe-count 2 \
  --probe-max-new-tokens 5000 \
  2>&1 | tee "$OUT/train.log"
