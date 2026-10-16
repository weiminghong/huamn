#!/usr/bin/env bash
set -euo pipefail

REPO=/home/weiminghong/humanslot
PYTHON=/home/weiminghong/yes/envs/human_v2/bin/python
SOURCE=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v5b_direct_joint_query_resume100_to500_gpu3/train/final-step-500
OUT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711/overfit8_v6_refpoint_pose_only_300steps_gpu3

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
  --resume-dir "$SOURCE" \
  --reset-optimizer-on-resume \
  --reset-pose-decoder-on-resume \
  --output-dir "$OUT/train" \
  --max-samples 8 \
  --epochs 150 \
  --max-optimizer-steps 300 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --max-length 5000 \
  --missing-attribute-loss-weight 0 \
  --ce-loss-weight 0 \
  --box-l1-loss-weight 0 \
  --box-giou-loss-weight 0 \
  --box-quality-loss-weight 0 \
  --pose-l1-loss-weight 5 \
  --pose-oks-loss-weight 1.5 \
  --pose-heatmap-loss-weight 0 \
  --pose-visibility-loss-weight 1 \
  --mask-bce-loss-weight 0 \
  --mask-dice-loss-weight 0 \
  --trainable-families pose \
  --num-workers 2 \
  --horizontal-flip-prob 0 \
  --base-lr 2e-5 \
  --visual-lr 5e-5 \
  --language-lr 2e-4 \
  --head-lr 1e-4 \
  --box-lr 5e-4 \
  --pose-lr 1e-3 \
  --mask-lr 1e-4 \
  --weight-decay 0 \
  --warmup-ratio 0.02 \
  --scheduler-type constant \
  --grad-clip 5 \
  --task-backbone-grad-scale 0 \
  --no-pose-box-jitter \
  --lora-r 8 \
  --lora-alpha 16 \
  --lora-dropout 0 \
  --decoder-dim 256 \
  --decoder-heads 8 \
  --bf16 \
  --seed 42 \
  --log-every 10 \
  --save-every 300 \
  --probe-every 0 \
  --probe-count 0 \
  2>&1 | tee "$OUT/train.log"
