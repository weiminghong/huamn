#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH=/root/autodl-tmp/humanslot
export HF_HOME=/root/autodl-tmp/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
BASE=/root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/exact_resume_deterministic_20260805
CONT="$BASE/continuous"
RES="$BASE/resumed"
cd /root/autodl-tmp/humanslot
echo "START $(date -Is)" > "$BASE/status.log"
COMMON=(--max-samples 80 --epochs 1 --max-optimizer-steps 5 --batch-size 4 --grad-accum-steps 4 --num-workers 0 --max-length 5000 --decoder-warmup-steps 2 --route-loss-hold-steps 1 --route-loss-warmup-steps 3 --save-every 3 --log-every 1 --seed 42 --deterministic)
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 --output-dir "$CONT" "${COMMON[@]}" > "$BASE/continuous.log" 2>&1
echo "CONTINUOUS_DONE $(date -Is)" >> "$BASE/status.log"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.train_stage1_v4 --output-dir "$RES" --resume-from "$CONT/step-3" "${COMMON[@]}" > "$BASE/resumed.log" 2>&1
echo "RESUME_DONE $(date -Is)" >> "$BASE/status.log"
/root/miniconda3/envs/human4b/bin/python -m human_lisa_qwen_v4.compare_checkpoints_v4 --first "$CONT/epoch-1-step-5" --second "$RES/epoch-1-step-5" --output "$BASE/comparison.json" > "$BASE/comparison.log" 2>&1
echo "SUCCESS $(date -Is)" >> "$BASE/status.log"
