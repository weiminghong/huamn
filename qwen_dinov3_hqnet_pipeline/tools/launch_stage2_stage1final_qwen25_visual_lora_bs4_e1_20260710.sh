#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_native_full_v1_20260710_stage1final_qwen25_visual_lora_bs4_e1
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4/train/final
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
{
  echo "started_at=$(date -Is)"
  echo "stage=stage2_train_qwen25_stage1final_dinov3_heads"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "qwen_model=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct"
  echo "qwen_adapter=${STAGE1}/base_or_adapter"
  echo "freeze_qwen=true"
  echo "train_dino=true"
  echo "batch_size=4"
  echo "head_lr=5e-5"
  echo "dino_lr=1e-6"
  echo "weight_decay=1e-4"
  echo "grad_clip=0.1"
  echo "epochs=1"
  echo "work_dir=${RUN}"
} | tee "$RUN/launch_status.log"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1/base_or_adapter" \
  --work-dir "$RUN" \
  --max-epochs 1 \
  --image-size 224 \
  --num-query 100 \
  --batch-size 4 \
  --lr 5e-5 \
  --dino-lr 1e-6 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --log-interval 20 \
  --save-interval 1000 \
  2>&1 | tee -a "$RUN/train.log"
{
  echo "finished_at=$(date -Is)"
  echo "stage=done_stage2_train"
  ls -lh "$RUN/latest.pt" 2>/dev/null || true
} | tee -a "$RUN/launch_status.log"
