#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
PY=/home/weiminghong/yes/envs/human_full/bin/python
RUN=/share2/program/weiminghong/humanslot_outputs/stage2_fullcfg_v1_20260713_stage1full_singlebackward_final16029_dinov3_hqnet300_img640_e1_acc4_hmwarmup_gpu2
STAGE1=/share2/program/weiminghong/humanslot_outputs/lisa_qwen_v1/stage1_v1_full_singlebackward_20260712_gpu3/train/final-step-16029
mkdir -p "$RUN"
export CUDA_VISIBLE_DEVICES=2
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}
{
  echo "started_at=$(date -Is)"
  echo "stage=stage2_train_fullcfg_hmwarmup_final16029_qwen25_dinov3_hqnet"
  echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
  echo "train_json=/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_train_v1.json"
  echo "train_image_root=/share2/program/weiminghong/coco/train2017"
  echo "val_json=/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json"
  echo "val_image_root=/share2/program/weiminghong/coco/val2017"
  echo "qwen_model=/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct"
  echo "qwen_adapter=${STAGE1}/base_or_adapter"
  echo "stage1_final=${STAGE1}"
  echo "profile=full"
  echo "image_size=640"
  echo "num_query=300"
  echo "batch_size=1"
  echo "grad_accum_steps=4"
  echo "effective_batch_size=4"
  echo "hm_loss_weight=0.004"
  echo "keypoint_clamp=true"
  echo "dynamic_heatmap_stride=true"
  echo "reopen_log_writer=true"
  echo "freeze_qwen=true"
  echo "train_dino=true"
  echo "head_lr=5e-5"
  echo "dino_lr=1e-6"
  echo "weight_decay=1e-4"
  echo "grad_clip=0.1"
  echo "epochs=1"
  echo "work_dir=${RUN}"
} | tee "$RUN/launch_status.log"
"$PY" qwen_dinov3_hqnet_pipeline/tools/train_stage2_qwen_dino_hqnet_native.py \
  --profile full \
  --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter "$STAGE1/base_or_adapter" \
  --work-dir "$RUN" \
  --max-epochs 1 \
  --image-size 640 \
  --num-query 300 \
  --batch-size 1 \
  --grad-accum-steps 4 \
  --hm-loss-weight 0.004 \
  --lr 5e-5 \
  --dino-lr 1e-6 \
  --weight-decay 1e-4 \
  --grad-clip 0.1 \
  --log-interval 20 \
  --save-interval 2000 \
  2>&1 | tee -a "$RUN/train.log"
{
  echo "train_finished_at=$(date -Is)"
  echo "stage=done_stage2_train_fullcfg_hmwarmup_final16029"
  ls -lh "$RUN/latest.pt" 2>/dev/null || true
} | tee -a "$RUN/launch_status.log"
if [ -f "$RUN/latest.pt" ]; then
  OUT="$RUN/eval_val_fullcfg_latest_gpu2"
  mkdir -p "$OUT"
  {
    echo "eval_started_at=$(date -Is)"
    echo "stage=eval_stage2_fullcfg_latest_val_final16029"
    echo "checkpoint=$RUN/latest.pt"
    echo "output_dir=$OUT"
    echo "profile=full"
    echo "image_size=640"
    echo "num_query=300"
    echo "score_thr=0.3"
    echo "official_score_thr=0.0"
  } | tee "$OUT/launch_status.log"
  "$PY" qwen_dinov3_hqnet_pipeline/tools/eval_stage2_qwen_dino_hqnet_native.py \
    --profile full \
    --checkpoint "$RUN/latest.pt" \
    --qwen-model /share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct \
    --qwen-adapter "$STAGE1/base_or_adapter" \
    --output-dir "$OUT" \
    --image-size 640 \
    --num-query 300 \
    --cuda-visible-devices 2 \
    --score-thr 0.3 \
    --official-score-thr 0.0 \
    --log-interval 20 \
    2>&1 | tee -a "$OUT/eval.log"
  {
    echo "eval_finished_at=$(date -Is)"
    echo "stage=done_eval_stage2_fullcfg_latest_val_final16029"
    ls -lh "$OUT/metrics.json" "$OUT/predictions.jsonl" 2>/dev/null || true
  } | tee -a "$OUT/launch_status.log"
fi
