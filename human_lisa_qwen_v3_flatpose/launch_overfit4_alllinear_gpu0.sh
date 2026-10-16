#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/root/autodl-tmp/humanslot
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v3_flatpose/gates/overfit4_alllinear_150steps_20260723
if [[ -e "$OUT" ]]; then
  echo "Refusing to overwrite $OUT" >&2
  exit 3
fi
mkdir -p "$(dirname "$OUT")"
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/train_stage1_v3.py   --output-dir "$OUT" --run-kind overfit --max-samples 4 --epochs 150   --max-optimizer-steps 150 --batch-size 4 --grad-accum-steps 1 --num-workers 4   --horizontal-flip-prob 0 --max-people 25 --person-query-layers 3   --pose-decoder-layers 2 --lora-target-mode all-linear   --scheduler-type constant --warmup-ratio 0.02   --visual-lr 1e-5 --language-lr 1e-5 --token-lr 2e-5 --head-lr 1e-3   --weight-decay 0.01 --grad-clip 5   --query-objectness-weight 1 --query-bbox-l1-weight 5 --query-giou-weight 2   --query-aux-weight 0.5 --pose-l1-weight 10 --pose-oks-weight 4   --pose-visibility-weight 1 --pose-aux-weight 0.5   --mask-bce-weight 2 --mask-dice-weight 5 --log-every 10   --no-save-epoch-checkpoints 2>&1 | tee "$OUT.train.log"
CKPT="$OUT/final-step-150"
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/eval_teacher_forced_v3.py   --checkpoint "$CKPT"   --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json   --image-root /root/autodl-tmp/coco/train2017   --output "$OUT/teacher_forced_metrics.json" --max-samples 4 --batch-size 4   2>&1 | tee "$OUT.teacher_eval.log"
/root/miniconda3/envs/human4b/bin/python -u human_lisa_qwen_v3_flatpose/eval_stage1_v3.py   --checkpoint "$CKPT"   --annotation /root/autodl-tmp/coco/unihuman_label/coco_unihuman_train_v1.json   --image-root /root/autodl-tmp/coco/train2017   --output-dir "$OUT/generated_train4" --max-samples 4 --batch-size 2 --max-new-tokens 5000   2>&1 | tee "$OUT.generated_train4.log"
