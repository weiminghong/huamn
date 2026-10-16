#!/usr/bin/env bash
set -euo pipefail

REPO=/root/autodl-tmp/humanslot
PY=/root/miniconda3/envs/human_stage2/bin/python
STAGE1_RUN=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_repr/full64115_2ep_bs8ga2_monitor_20260728
STAGE1_ADAPTER=${STAGE1_RUN}/epoch-1-step-4008/base_or_adapter
QWEN=/root/autodl-tmp/model/Qwen3.5-4B
DINO=/root/autodl-tmp/model/dinov3-vits16-pretrain-lvd1689m-timm
TRAIN_ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json
TRAIN_IMG=/root/autodl-tmp/coco/train2017
VAL_ANN=/root/autodl-tmp/coco/unihuman_label/coco_unihuman_val_v1.json
VAL_IMG=/root/autodl-tmp/coco/val2017
ROOT=/root/autodl-tmp/humanslot_outputs/stage2_epoch1_full_pipeline_20260729
SETUP_LOG=${REPO}/qwen_dinov3_hqnet_pipeline/tools/setup_stage2_env_a800_20260729.log
STATUS=${ROOT}/status.log

mkdir -p "$ROOT"
exec > >(tee -a "$STATUS") 2>&1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO:$REPO/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}"
export HF_HOME=/root/autodl-tmp/hf_stage2
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_ENDPOINT=https://hf-mirror.com

status() { echo "[$(date '+%F %T %Z')] $*"; }

status "queued: full Stage2 pipeline only; no DINO-only branch"
if ! grep -q 'setup complete' "$SETUP_LOG"; then
  status "ERROR: validated Stage2 environment completion marker missing"
  exit 2
fi
if [[ ! -f "$STAGE1_ADAPTER/adapter_model.safetensors" ]]; then
  status "ERROR: Epoch1 Stage1 adapter missing"
  exit 3
fi

status "waiting for Stage1 epoch1 full-val evaluator to release GPU0"
while pgrep -af 'human_lisa_qwen_v3_repr/eval_stage1_v3.py' | grep -q "$STAGE1_RUN"; do
  sleep 60
done
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if (( FREE_MIB >= 78000 )); then
    status "GPU0 released; free_mib=$FREE_MIB"
    break
  fi
  status "GPU0 still occupied; free_mib=$FREE_MIB; waiting"
  sleep 60
done

cd "$REPO"
status "preflight tests"
"$PY" qwen_dinov3_hqnet_pipeline/tests/test_stage2_static.py
"$PY" -m pytest -q qwen_dinov3_hqnet_pipeline/tests/test_stage2_official_export.py

COMMON=(
  --backbone-mode qwen_fusion
  --qwen-model "$QWEN"
  --qwen-adapter "$STAGE1_ADAPTER"
  --dino-model "$DINO"
  --fusion-mode feature_add
  --train-ann-file "$TRAIN_ANN"
  --train-image-root "$TRAIN_IMG"
  --profile full
  --image-size 640
  --num-query 300
  --max-instances 25
  --max-epochs 1
  --bbox-loss-weight 1
  --iou-loss-weight 1
  --cls-loss-weight 0.05
  --kpt-cls-loss-weight 0.05
  --hm-loss-weight 0.001
  --kpt-loss-weight 2
  --kpt-rpn-loss-weight 2
  --oks-loss-weight 0.2
  --kpt-refine-loss-weight 2
  --oks-refine-loss-weight 0.2
  --mask-loss-weight 0.5
  --dice-loss-weight 0.2
  --lr 5e-5
  --dino-lr 1e-6
  --weight-decay 1e-4
  --grad-clip 0.1
  --seed 42
)

status "full-profile memory/loss smoke start: batch=2, steps=3"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --batch-size 2 --grad-accum-steps 1 --max-steps 3 --limit-images 16 \
  --work-dir "$ROOT/smoke_full_pipeline_bs2" --log-interval 1 --save-interval 3 \
  2>&1 | tee "$ROOT/smoke_full_pipeline_bs2.log"
status "full-profile smoke passed"
rm -f -- "$ROOT/smoke_full_pipeline_bs2"/latest.pt "$ROOT/smoke_full_pipeline_bs2"/checkpoint_step_*.pt

status "formal full-train epoch start: 64115 images, batch=2, grad_accum=2, effective_batch=4"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --batch-size 2 --grad-accum-steps 2 --max-steps 0 \
  --work-dir "$ROOT/train_full_epoch1" --log-interval 20 --save-interval 1000 \
  2>&1 | tee "$ROOT/train_full_epoch1.log"

status "full 2693-image val evaluation start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
  --backbone-mode qwen_fusion --qwen-model "$QWEN" --qwen-adapter "$STAGE1_ADAPTER" \
  --dino-model "$DINO" --fusion-mode feature_add \
  --checkpoint "$ROOT/train_full_epoch1/latest.pt" \
  --val-ann-file "$VAL_ANN" --val-image-root "$VAL_IMG" \
  --output-dir "$ROOT/eval_full_epoch1_val2693" \
  --profile full --image-size 640 --num-query 300 --batch-size 1 --official-score-thr 0 \
  --cuda-visible-devices 0 --log-interval 50 \
  2>&1 | tee "$ROOT/eval_full_epoch1_val2693.log"

"$PY" - <<PY
import json
from pathlib import Path
root=Path('$ROOT')
m=json.load(open(root/'eval_full_epoch1_val2693'/'metrics.json'))
o=m.get('official_eval',{})
out={
 'protocol':'stage1_epoch1_frozen_qwen_plus_trainable_dinov3_hqnet_full_train_epoch1_fullval2693',
 'classification':'diagnostic_native_stage2_evaluator',
 'images':m.get('images'),'errors':m.get('errors'),'gt_persons':m.get('gt_persons'),'pred_persons':m.get('pred_persons'),
 'bbox_AP':o.get('bbox_AP'),'bbox_AP50':o.get('bbox_AP50'),'bbox_AP75':o.get('bbox_AP75'),
 'keypoints_AP':o.get('keypoints_AP'),'keypoints_AP50':o.get('keypoints_AP50'),
 'segm_AP':o.get('segm_AP'),'segm_AP50':o.get('segm_AP50'),
 'gender_AP50':o.get('gender_AP50'),'age_AP50':o.get('age_AP50'),
}
(root/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
PY
status "full Stage2 pipeline trial complete"
