#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_fullcfg_v1_20260711_stage1final_qwen25_dinov3_hqnet300_img640_e1_acc4
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_epoch1_20260709_qwen25_visual_lora_effbs4/train/final
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
{
  echo "started_at=$(date -Is)"
  echo "stage=stage2_train_fullcfg_stage1final_qwen25_dinov3_hqnet"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "train_json=/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json"
  echo "train_image_root=/share2/program/weiminghong/coco/train2017"
  echo "qwen_model=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct"
  echo "qwen_adapter=${STAGE1}/base_or_adapter"
  echo "profile=full"
  echo "image_size=640"
  echo "num_query=300"
  echo "batch_size=1"
  echo "grad_accum_steps=4"
  echo "effective_batch_size=4"
  echo "freeze_qwen=true"
  echo "train_dino=true"
  echo "head_lr=5e-5"
  echo "dino_lr=1e-6"
  echo "weight_decay=1e-4"
  echo "grad_clip=0.1"
  echo "epochs=1"
  echo "work_dir=${RUN}"
} | tee "$RUN/launch_status.log"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  --profile full \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1/base_or_adapter" \
  --work-dir "$RUN" \
  --max-epochs 1 \
  --image-size 640 \
  --num-query 300 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --lr 5e-5 \
  --dino-lr 1e-6 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --log-interval 20 \
  --save-interval 500 \
  2>&1 | tee -a "$RUN/train.log"
{
  echo "finished_at=$(date -Is)"
  echo "stage=done_stage2_train_fullcfg"
  ls -lh "$RUN/latest.pt" 2>/dev/null || true
} | tee -a "$RUN/launch_status.log"
