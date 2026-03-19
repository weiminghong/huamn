cd ../..

CONFIG_FILE="configs/qwen_vl_stage1.py"
CHECKPOINT="/path/to/stage1_checkpoint"
SPLIT="val"

python tools/test.py \
  --config ${CONFIG_FILE} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT}
