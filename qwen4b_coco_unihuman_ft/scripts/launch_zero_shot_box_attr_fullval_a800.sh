#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/hf_cache/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hf_cache/transformers
export CUDA_VISIBLE_DEVICES=0
PY=/root/miniconda3/envs/human4b/bin/python
OUT=/root/autodl-tmp/humanslot_outputs/qwen35_zero_shot_box_attr_v1/fullval2693_batch2_20260722
mkdir -p "$OUT"
cat > "$OUT/status.log" <<STATUS
started_at=$(date -Is)
protocol=native_qwen35_zero_shot_guided_json_box_gender_age
classification=diagnostic_until_score_protocol_fixed
batch_size=2
max_new_tokens=1024
max_samples=2693
model=/root/autodl-tmp/model/Qwen3.5-4B
annotation=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
image_root=/root/autodl-tmp/coco/val2017
STATUS
finish() {
  code=$?
  trap - EXIT
  echo "finished_at=$(date -Is)" >> "$OUT/status.log"
  echo "exit_code=$code" >> "$OUT/status.log"
  if [ "$code" -eq 0 ]; then echo status=completed >> "$OUT/status.log"; else echo status=failed >> "$OUT/status.log"; fi
  exit "$code"
}
trap finish EXIT
"$PY" -u qwen4b_coco_unihuman_ft/eval_zero_shot_box_attr_qwen35.py \
  --model /root/autodl-tmp/model/Qwen3.5-4B \
  --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json \
  --image-root /root/autodl-tmp/coco/val2017 \
  --output-dir "$OUT" \
  --max-samples 2693 \
  --batch-size 2 \
  --max-new-tokens 1024 \
  --cuda-visible-devices 0 \
  --resume \
  2>&1 | tee -a "$OUT/run.log"
