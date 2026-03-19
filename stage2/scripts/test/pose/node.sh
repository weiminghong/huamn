cd ../../..

# DEVICES=0,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_test.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

CONFIG_FILE=configs/pose/human_pose_sapiens_1b-210e-1024x768.py
CHECKPOINT='/path/to/your_pose_checkpoint.pth'
TEST_BATCH_SIZE_PER_GPU=32
OUTPUT_DIR="Outputs/test/pose/node/$(date +"%m-%d-%Y_%H:%M:%S")"

mode='multi-gpu'

if [ "$mode" = "debug" ]; then
    OPTIONS="test_dataloader.batch_size=4 test_dataloader.num_workers=0 test_dataloader.persistent_workers=False"
    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/test.py ${CONFIG_FILE} ${CHECKPOINT} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}
else
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    OPTIONS="test_dataloader.batch_size=${TEST_BATCH_SIZE_PER_GPU}"
    LOG_FILE="${OUTPUT_DIR}/log.txt"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} ${CHECKPOINT} \
        ${NUM_GPUS} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options ${OPTIONS} \
        | tee ${LOG_FILE}
fi
