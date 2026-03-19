cd ../../..

# DEVICES=0,
DEVICES=0,1,2,3,4,5,6,7,

RUN_FILE='./tools/dist_train.sh'
PORT=$(( ((RANDOM<<15)|RANDOM) % 63001 + 2000 ))

CONFIG_FILE=configs/parsing/human_parsing_sapiens_1b-200e-1024x768.py
TRAIN_BATCH_SIZE_PER_GPU=2
OUTPUT_DIR="Outputs/train/parsing/node/$(date +"%m-%d-%Y_%H:%M:%S")"

RESUME_FROM=''
LOAD_FROM=''
mode='multi-gpu'

if [ -n "$LOAD_FROM" ]; then
    OPTIONS="train_dataloader.batch_size=${TRAIN_BATCH_SIZE_PER_GPU} load_from=${LOAD_FROM}"
else
    OPTIONS="train_dataloader.batch_size=${TRAIN_BATCH_SIZE_PER_GPU}"
fi

if [ -n "$RESUME_FROM" ]; then
    CMD_RESUME="--resume ${RESUME_FROM}"
else
    CMD_RESUME=""
fi

if [ "$mode" = "debug" ]; then
    OPTIONS="train_dataloader.batch_size=1 train_dataloader.num_workers=0 train_dataloader.persistent_workers=False"
    CUDA_VISIBLE_DEVICES=${DEVICES} python tools/train.py ${CONFIG_FILE} --work-dir ${OUTPUT_DIR} --cfg-options ${OPTIONS}
else
    NUM_GPUS_STRING_LEN=${#DEVICES}
    NUM_GPUS=$((NUM_GPUS_STRING_LEN/2))
    LOG_FILE="${OUTPUT_DIR}/log.txt"
    mkdir -p ${OUTPUT_DIR}; touch ${LOG_FILE}

    CUDA_VISIBLE_DEVICES=${DEVICES} PORT=${PORT} ${RUN_FILE} ${CONFIG_FILE} \
        ${NUM_GPUS} \
        --work-dir ${OUTPUT_DIR} \
        --cfg-options ${OPTIONS} \
        ${CMD_RESUME} \
        | tee ${LOG_FILE}
fi
