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
ROOT=/root/autodl-tmp/humanslot_outputs/stage2_epoch1_pair_20260729
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

status "queued: waiting for independent Stage2 environment"
while screen -ls 2>/dev/null | grep -q '[.]stage2_setup'; do sleep 60; done
if ! grep -q 'setup complete' "$SETUP_LOG"; then
  status "ERROR: Stage2 setup exited without completion marker"
  exit 2
fi

status "environment ready; waiting for Stage1 trainer to release GPU0"
while pgrep -af 'human_lisa_qwen_v3_repr/train_stage1_v3.py' | grep -q "$STAGE1_RUN"; do sleep 60; done
if [[ ! -f "$STAGE1_ADAPTER/adapter_model.safetensors" ]]; then
  status "ERROR: Epoch1 adapter missing"
  exit 3
fi
while true; do
  FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
  if (( FREE_MIB >= 78000 )); then
    status "GPU fully released (no Stage1 evaluator footprint); free_mib=$FREE_MIB"
    break
  fi
  status "GPU still occupied by Stage1 post-evaluation; free_mib=$FREE_MIB; waiting"
  sleep 60
done

cd "$REPO"
status "preflight tests"
"$PY" qwen_dinov3_hqnet_pipeline/tests/test_stage2_static.py
"$PY" -m pytest -q qwen_dinov3_hqnet_pipeline/tests/test_stage2_official_export.py

COMMON=(
  --train-ann-file "$TRAIN_ANN"
  --train-image-root "$TRAIN_IMG"
  --qwen-model "$QWEN"
  --dino-model "$DINO"
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

SMOKE_INIT=${ROOT}/smoke_shared_initial_state.pt
status "smoke baseline start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --backbone-mode dino_only --profile lite --image-size 224 --num-query 16 \
  --batch-size 1 --grad-accum-steps 1 --max-steps 5 --limit-images 32 --max-instances 4 \
  --shared-init-state "$SMOKE_INIT" --write-shared-init \
  --work-dir "$ROOT/smoke_dino_only" --log-interval 1 --save-interval 5 \
  2>&1 | tee "$ROOT/smoke_dino_only.log"

status "smoke qwen fusion start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --backbone-mode qwen_fusion --qwen-adapter "$STAGE1_ADAPTER" \
  --fusion-mode feature_add --profile lite --image-size 224 --num-query 16 \
  --batch-size 1 --grad-accum-steps 1 --max-steps 5 --limit-images 32 --max-instances 4 \
  --shared-init-state "$SMOKE_INIT" \
  --work-dir "$ROOT/smoke_qwen_fusion" --log-interval 1 --save-interval 5 \
  2>&1 | tee "$ROOT/smoke_qwen_fusion.log"

FULL_INIT=${ROOT}/full_shared_initial_state.pt
status "paired baseline 1000-step training start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --backbone-mode dino_only --profile full --image-size 640 --num-query 300 \
  --batch-size 1 --grad-accum-steps 4 --max-steps 1000 \
  --shared-init-state "$FULL_INIT" --write-shared-init \
  --work-dir "$ROOT/train_dino_only_1000" --log-interval 20 --save-interval 500 \
  2>&1 | tee "$ROOT/train_dino_only_1000.log"

status "paired Epoch1-Qwen fusion 1000-step training start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  "${COMMON[@]}" --backbone-mode qwen_fusion --qwen-adapter "$STAGE1_ADAPTER" \
  --fusion-mode feature_add --profile full --image-size 640 --num-query 300 \
  --batch-size 1 --grad-accum-steps 4 --max-steps 1000 \
  --shared-init-state "$FULL_INIT" \
  --work-dir "$ROOT/train_qwen_fusion_1000" --log-interval 20 --save-interval 500 \
  2>&1 | tee "$ROOT/train_qwen_fusion_1000.log"

status "full-val DINO-only evaluation start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
  --backbone-mode dino_only --qwen-model "$QWEN" --qwen-adapter '' --dino-model "$DINO" \
  --checkpoint "$ROOT/train_dino_only_1000/latest.pt" \
  --val-ann-file "$VAL_ANN" --val-image-root "$VAL_IMG" \
  --output-dir "$ROOT/eval_dino_only_1000_fullval2693" \
  --profile full --image-size 640 --num-query 300 --batch-size 1 --official-score-thr 0 \
  --cuda-visible-devices 0 --log-interval 50 \
  2>&1 | tee "$ROOT/eval_dino_only_1000_fullval2693.log"

status "full-val Epoch1-Qwen fusion evaluation start"
"$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
  --backbone-mode qwen_fusion --qwen-model "$QWEN" --qwen-adapter "$STAGE1_ADAPTER" \
  --dino-model "$DINO" --fusion-mode feature_add \
  --checkpoint "$ROOT/train_qwen_fusion_1000/latest.pt" \
  --val-ann-file "$VAL_ANN" --val-image-root "$VAL_IMG" \
  --output-dir "$ROOT/eval_qwen_fusion_1000_fullval2693" \
  --profile full --image-size 640 --num-query 300 --batch-size 1 --official-score-thr 0 \
  --cuda-visible-devices 0 --log-interval 50 \
  2>&1 | tee "$ROOT/eval_qwen_fusion_1000_fullval2693.log"

"$PY" - <<PY
import json, pathlib
root=pathlib.Path('$ROOT')
paths={
 'dino_only': root/'eval_dino_only_1000_fullval2693'/'metrics.json',
 'qwen_fusion': root/'eval_qwen_fusion_1000_fullval2693'/'metrics.json',
}
out={'protocol':'paired_1000step_diagnostic_fullval2693','shared_initialization':str(root/'full_shared_initial_state.pt'),'results':{}}
for name,path in paths.items():
 m=json.load(open(path))
 official=m.get('official_eval',{})
 out['results'][name]={
  'images':m.get('images'),'errors':m.get('errors'),'gt_persons':m.get('gt_persons'),'pred_persons':m.get('pred_persons'),
  'bbox_AP50':official.get('bbox_AP50'),'keypoints_AP':official.get('keypoints_AP'),'keypoints_AP50':official.get('keypoints_AP50'),
  'segm_AP':official.get('segm_AP'),'segm_AP50':official.get('segm_AP50'),'gender_AP50':official.get('gender_AP50'),'age_AP50':official.get('age_AP50'),
 }
a=out['results']['dino_only']; b=out['results']['qwen_fusion']
out['qwen_minus_dino']={k:(b[k]-a[k] if isinstance(a.get(k),(int,float)) and isinstance(b.get(k),(int,float)) else None) for k in sorted(set(a)|set(b))}
(root/'comparison.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
PY
status "paired Stage2 trial complete"
