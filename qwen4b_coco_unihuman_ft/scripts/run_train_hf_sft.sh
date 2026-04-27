#!/usr/bin/env bash
set -euo pipefail

python qwen4b_coco_unihuman_ft/train_sft.py \
  --config qwen4b_coco_unihuman_ft/configs/qwen_vl_box_pose_attr_sft_server.yaml
