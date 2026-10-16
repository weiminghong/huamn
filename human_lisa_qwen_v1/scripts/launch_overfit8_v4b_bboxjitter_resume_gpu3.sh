#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/weiminghong/yes/envs/human_v2/bin/python
ROOT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_rebuild_20260711
SOURCE="$ROOT/overfit8_v4_topdown_heatmap_100steps_gpu3/train/final-step-100"
RUN="$ROOT/overfit8_v4b_bboxjitter_resume100_to200_gpu3"
test -f "$SOURCE/manifest.json"
test ! -e "$RUN"
mkdir -p "$RUN"
{
  echo "started_at=$(date -Is)"
  echo "source=$SOURCE"
  echo "architecture=topdown_roi_heatmap_softargmax"
  echo "pose_box_jitter=true"
  echo "pose_box_shift_factor=0.16"
  echo "pose_box_shift_prob=0.3"
  echo "pose_box_scale_range=0.5,1.5"
  echo "pose_box_scale_prob=1.0"
} > "$RUN/status.log"
finish() {
  code=$?
  trap - EXIT
  echo "finished_at=$(date -Is)" >> "$RUN/status.log"
  echo "exit_code=$code" >> "$RUN/status.log"
  if [ "$code" -eq 0 ]; then
    echo "status=completed" >> "$RUN/status.log"
  else
    echo "status=failed" >> "$RUN/status.log"
  fi
  exit "$code"
}
trap finish EXIT
"$PY" -u human_lisa_qwen_v1/train_stage1_v1.py \
  --resume-dir "$SOURCE" --output-dir "$RUN/train" \
  --max-samples 8 --epochs 100 --max-optimizer-steps 200 \
  --batch-size 1 --grad-accum-steps 4 --max-length 5000 \
  --num-workers 0 --horizontal-flip-prob 0 \
  --visual-lr 5e-5 --language-lr 2e-4 --head-lr 1e-4 \
  --box-lr 5e-4 --pose-lr 1e-3 --mask-lr 1e-4 \
  --scheduler-type constant --warmup-ratio 0.02 --grad-clip 5 \
  --task-backbone-grad-scale 0.1 --lora-dropout 0 \
  --pose-box-jitter --pose-box-shift-factor 0.16 \
  --pose-box-shift-prob 0.3 --pose-box-scale-min 0.5 \
  --pose-box-scale-max 1.5 --pose-box-scale-prob 1.0 \
  --log-every 10 --save-every 0 --probe-every 200 --probe-count 4 \
  --probe-max-new-tokens 5000 --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/train.log"
