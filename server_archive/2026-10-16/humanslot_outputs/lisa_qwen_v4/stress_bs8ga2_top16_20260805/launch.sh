#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/root/autodl-tmp/humanslot
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/stress_bs8ga2_top16_20260805
cd /root/autodl-tmp/humanslot
echo "START $(date -Is)" > "$BASE/status.log"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 \
 --train-ann /root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/stress_top16_train_v1_20260805.json \
 --train-image-root /root/autodl-tmp/coco/train2017 \
 --model /root/autodl-tmp/model/Qwen3.5-4B \
 --output-dir "$BASE/run" --profile vision_tuned --epochs 1 \
 --max-optimizer-steps 1 --batch-size 8 --grad-accum-steps 2 \
 --num-workers 4 --max-length 5000 --save-every 0 --log-every 1 \
 --seed 42 --deterministic > "$BASE/train.log" 2>&1
echo "SUCCESS $(date -Is)" >> "$BASE/status.log"
