#!/usr/bin/env bash
set -euo pipefail

python qwen4b_coco_unihuman_ft/convert_coco_unihuman.py \
  --train-ann-json /home/priv/wmh_project/coco_person/labels/coco_unihuman_train_v1.json \
  --val-ann-json /home/priv/wmh_project/coco_person/labels/coco_unihuman_val_v1.json \
  --image-root /home/priv/wmh_project/coco_person/images \
  --output-dir /home/priv/wmh_project/coco_person/ft_box_pose_attr_data \
  --emit-swift-format

echo "Done. Output dir: /home/priv/wmh_project/coco_person/ft_box_pose_attr_data"
