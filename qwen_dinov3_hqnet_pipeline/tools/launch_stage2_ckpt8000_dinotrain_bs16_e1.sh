#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260709_ckpt8000_dinotrain_bs16_e1
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
{
  echo "started_at=$(date -Is)"
  echo "host=$(hostname)"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "qwen_adapter=/home/weiminghong/humanslot/outputs/hsm_stage1_teacher/legacy_sft/checkpoint-8000"
  echo "work_dir=${RUN}"
} > "$RUN/launch_manifest.txt"
exec "$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  --qwen-adapter /home/weiminghong/humanslot/outputs/hsm_stage1_teacher/legacy_sft/checkpoint-8000 \
  --work-dir "$RUN" \
  --max-epochs 1 \
  --image-size 224 \
  --num-query 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --dino-lr 1e-5 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --log-interval 20 \
  --save-interval 500 \
  2>&1 | tee -a "$RUN/train.log"
