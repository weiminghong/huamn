#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
GPU=${GPU:-2}
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/share2/program/weiminghong/humanslot_outputs/schema_decode_pkgs:/home/weiminghong/humanslot
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/home/weiminghong/yes/envs/human_v2/bin/python
ROOT=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1
RUN="$ROOT/stage1_v1_stable5task_full_3epoch_$(date +%Y%m%d_%H%M%S)_gpu${GPU}"
if [ -e "$RUN" ]; then
  echo "refusing to overwrite $RUN" >&2
  exit 2
fi
mkdir -p "$RUN"
cat > "$RUN/status.log" <<STATUS
started_at=$(date -Is)
gpu=$GPU
purpose=full_stage1_after_stability_probe
protocol=COCO-UniHuman-v1-five-task
epochs=3
effective_batch=4
max_length=5000
split_ce_task_backward=true
language_lr=1e-5
visual_lr=1e-5
head_lr=5e-5
box_lr=5e-5
pose_lr=5e-5
mask_lr=2e-5
task_backbone_grad_scale=0.02
mask_loss_weights=bce4_dice2
STATUS
finish() {
  code=$?
  trap - EXIT
  echo "finished_at=$(date -Is)" >> "$RUN/status.log"
  echo "exit_code=$code" >> "$RUN/status.log"
  if [ "$code" -eq 0 ]; then echo status=completed >> "$RUN/status.log"; else echo status=failed >> "$RUN/status.log"; fi
  exit "$code"
}
trap finish EXIT
"$PY" -u human_lisa_qwen_v1/train_stage1_v1.py \
  --output-dir "$RUN/train" \
  --epochs 3 \
  --batch-size 1 --grad-accum-steps 4 --max-length 5000 \
  --num-workers 2 --horizontal-flip-prob 0.5 \
  --base-lr 1e-5 --visual-lr 1e-5 --language-lr 1e-5 --head-lr 5e-5 \
  --box-lr 5e-5 --pose-lr 5e-5 --mask-lr 2e-5 \
  --scheduler-type cosine --warmup-ratio 0.05 --grad-clip 1 \
  --task-backbone-grad-scale 0.02 --lora-dropout 0.05 \
  --split-ce-task-backward \
  --no-lora-embed-tokens --lora-lm-head \
  --ce-loss-weight 1 \
  --box-l1-loss-weight 5 --box-giou-loss-weight 2 --box-quality-loss-weight 1 \
  --pose-l1-loss-weight 5 --pose-oks-loss-weight 1.5 --pose-visibility-loss-weight 1 \
  --mask-bce-loss-weight 4 --mask-dice-loss-weight 2 \
  --missing-attribute-loss-weight 0 \
  --log-every 10 --save-every 2000 --probe-every 1000 --probe-count 4 \
  --probe-max-new-tokens 5000 --bf16 --cuda-visible-devices "$GPU" \
  2>&1 | tee "$RUN/train.log"
