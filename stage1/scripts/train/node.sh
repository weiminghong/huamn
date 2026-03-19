cd ../..

CONFIG_FILE="configs/qwen_vl_stage1.py"
OUTPUT_DIR="Outputs/stage1/qwen_vl_3b/node/$(date +"%m-%d-%Y_%H:%M:%S")"

python tools/train.py \
  --config ${CONFIG_FILE} \
  --output-dir ${OUTPUT_DIR}
