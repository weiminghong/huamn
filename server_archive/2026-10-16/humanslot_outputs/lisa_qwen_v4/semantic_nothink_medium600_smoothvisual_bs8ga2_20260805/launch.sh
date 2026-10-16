#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/semantic_nothink_medium600_smoothvisual_bs8ga2_20260805
mkdir -p "$BASE"
cd /root/autodl-tmp/humanslot
status=1
finish() {
  if [ "$status" -eq 0 ]; then
    echo "SUCCESS $(date -Is)" >> "$BASE/status.log"
  else
    echo "FAILED exit=$status $(date -Is)" >> "$BASE/status.log"
  fi
}
trap 'status=$?; finish' EXIT
echo "START $(date -Is)" > "$BASE/status.log"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4   --train-ann /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json   --train-image-root /root/autodl-tmp/coco/train2017   --model /root/autodl-tmp/model/Qwen3.5-4B   --output-dir "$BASE/run" --profile vision_tuned --epochs 2   --stop-after-optimizer-steps 600 --batch-size 8 --grad-accum-steps 2   --num-workers 8 --max-length 5000 --save-every 100 --log-every 10   --token-init semantic --decoder-warmup-steps 200 --visual-warmup-steps 200   --seed 42 --deterministic > "$BASE/train.log" 2>&1
status=0
