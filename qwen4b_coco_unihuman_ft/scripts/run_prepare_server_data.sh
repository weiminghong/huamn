#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-/home/priv/wmh_project/coco_person}
TRAIN_IMAGE_ROOT=${TRAIN_IMAGE_ROOT:-${DATA_ROOT}/images}
VAL_IMAGE_ROOT=${VAL_IMAGE_ROOT:-${DATA_ROOT}/images}
OUTPUT_DIR=${OUTPUT_DIR:-${DATA_ROOT}/ft_box_pose_attr_data}

python qwen4b_coco_unihuman_ft/convert_coco_unihuman.py \
  --train-ann-json ${DATA_ROOT}/labels/coco_unihuman_train_v1.json \
  --val-ann-json ${DATA_ROOT}/labels/coco_unihuman_val_v1.json \
  --image-root ${TRAIN_IMAGE_ROOT} \
  --train-image-root ${TRAIN_IMAGE_ROOT} \
  --val-image-root ${VAL_IMAGE_ROOT} \
  --output-dir ${OUTPUT_DIR} \
  --emit-swift-format

echo "Done. Output dir: ${OUTPUT_DIR}"
