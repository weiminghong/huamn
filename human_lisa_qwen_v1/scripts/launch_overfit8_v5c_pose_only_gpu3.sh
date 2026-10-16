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
SOURCE="$ROOT/overfit8_v5b_direct_joint_query_resume100_to500_gpu3/train/final-step-500"
RUN="$ROOT/overfit8_v5c_pose_only_resume500_to800_gpu3"
test -f "$SOURCE/manifest.json"
test ! -e "$RUN"
mkdir -p "$RUN"
{
  echo "started_at=$(date -Is)"
  echo "source=$SOURCE"
  echo "diagnostic=pose_only"
  echo "pose_decoder=direct_joint_query_full_image"
  echo "trainable_effective=pose_decoder_only"
  echo "task_backbone_grad_scale=0"
  echo "pose_loss=l1+oks+visibility"
  echo "pose_box_jitter=false"
  echo "pose_roi=false"
  echo "pose_heatmap=false"
  echo "target_optimizer_steps=800"
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
  --max-samples 8 --epochs 400 --max-optimizer-steps 800 \
  --batch-size 1 --grad-accum-steps 4 --max-length 5000 \
  --num-workers 0 --horizontal-flip-prob 0 \
  --visual-lr 5e-5 --language-lr 2e-4 --head-lr 1e-4 \
  --box-lr 5e-4 --pose-lr 1e-3 --mask-lr 1e-4 \
  --scheduler-type constant --warmup-ratio 0.02 --grad-clip 5 \
  --task-backbone-grad-scale 0 --lora-dropout 0 \
  --no-pose-box-jitter --missing-attribute-loss-weight 0 \
  --ce-loss-weight 0 \
  --box-l1-loss-weight 0 --box-giou-loss-weight 0 \
  --box-quality-loss-weight 0 \
  --pose-l1-loss-weight 5 --pose-oks-loss-weight 1.5 \
  --pose-heatmap-loss-weight 0 --pose-visibility-loss-weight 1 \
  --mask-bce-loss-weight 0 --mask-dice-loss-weight 0 \
  --log-every 10 --save-every 0 --probe-every 0 --probe-count 0 \
  --bf16 --cuda-visible-devices 3 \
  2>&1 | tee "$RUN/train.log"

