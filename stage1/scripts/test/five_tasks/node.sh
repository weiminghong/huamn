cd ../../..

CONFIG_FILE="configs/qwen_vl_stage1_five_tasks.py"
CHECKPOINT="/path/to/stage1_five_tasks_checkpoint"
SPLIT="val"

python tools/test_five_tasks.py \
  --config ${CONFIG_FILE} \
  --checkpoint ${CHECKPOINT} \
  --split ${SPLIT}
