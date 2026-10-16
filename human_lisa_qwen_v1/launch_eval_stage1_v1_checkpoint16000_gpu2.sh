#!/usr/bin/env bash
set -euo pipefail
REPO=/home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_v2/bin/python
CKPT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_joint_vlora5task_3epoch_20260714_gpu2/train/checkpoint-16000
OUT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_joint_vlora5task_3epoch_20260714_gpu2/eval_val_official_checkpoint-16000_gpu2
LOG=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_joint_vlora5task_3epoch_20260714_gpu2/eval_val_official_checkpoint-16000_gpu2.log
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/share2/program/weiminghong/hf_cache
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
mkdir -p "$OUT"
cd "$REPO"
echo "[eval] checkpoint=$CKPT"
echo "[eval] output=$OUT"
"$PY" -u human_lisa_qwen_v1/eval_stage1_v1.py   --annotation /share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json   --image-root /share2/program/weiminghong/coco/val2017   --checkpoint "$CKPT"   --output-dir "$OUT"   --max-new-tokens 5000   --batch-size 1   --bf16   2>&1 | tee "$LOG"
