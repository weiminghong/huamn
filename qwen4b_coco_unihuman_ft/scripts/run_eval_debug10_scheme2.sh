#!/usr/bin/env bash
set -euo pipefail

python qwen4b_coco_unihuman_ft/infer_eval_vis.py \
  --config qwen4b_coco_unihuman_ft/configs/eval_box_pose_attr_debug10_server.yaml
