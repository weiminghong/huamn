#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MAX_PIXELS=${MAX_PIXELS:-1003520}

swift sft \
  --model ${MODEL_PATH:-/home/priv/project/model/Qwen2.5-VL-3B-Instruct} \
  --dataset /home/priv/wmh_project/coco_person/ft_box_pose_attr_data/train_swift.jsonl \
  --val_dataset /home/priv/wmh_project/coco_person/ft_box_pose_attr_data/val_swift.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --num_train_epochs 2 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --learning_rate 1e-4 \
  --lora_rank 8 \
  --lora_alpha 32 \
  --target_modules all-linear \
  --gradient_accumulation_steps 16 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 2 \
  --logging_steps 10 \
  --max_length 4096 \
  --output_dir /home/priv/wmh_project/coco_person/outputs/qwen_vl_box_pose_attr_lora \
  --warmup_ratio 0.05 \
  --dataloader_num_workers 4
