cd ../../..

CONFIG_FILE="configs/qwen_vl_stage1_five_tasks.py"
OUTPUT_DIR="Outputs/stage1/qwen_vl_3b_five_tasks/node/$(date +"%m-%d-%Y_%H:%M:%S")"

python tools/train_five_tasks.py \
  --config ${CONFIG_FILE} \
  --output-dir ${OUTPUT_DIR}
