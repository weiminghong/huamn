#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_balancedloss_v3_accum1_20260713_stage1full_final16029_gpu1_max50
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/train/final-step-16029
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
echo "launch_time_utc=$(date -u '+%Y-%m-%d %H:%M:%S UTC') accum=1 max_steps=50" | tee -a "$RUN/launch_status.txt"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  --profile full \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1/base_or_adapter" \
  --work-dir "$RUN" \
  --max-epochs 1 \
  --max-steps 50 \
  --image-size 640 \
  --num-query 300 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --cls-loss-weight 0.25 \
  --kpt-cls-loss-weight 0.25 \
  --hm-loss-weight 0.004 \
  --kpt-loss-weight 10 \
  --kpt-rpn-loss-weight 10 \
  --oks-loss-weight 1 \
  --kpt-refine-loss-weight 10 \
  --oks-refine-loss-weight 1 \
  --mask-loss-weight 2 \
  --dice-loss-weight 1 \
  --lr 5e-5 \
  --dino-lr 1e-6 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --log-interval 10 \
  --save-interval 0 \
  2>&1 | tee -a "$RUN/train.log"
