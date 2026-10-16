#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export GPU=2
export BS=1
export MAX_NEW_TOKENS=1024
export MAX_SAMPLES=999999
export CKPTS="final"
export OUT_TAG="gpu2_bs1_tok1024_strict"
bash human_lisa_qwen_v1/launch_eval_ckpt56000_48000_val_gpu3_20260710.sh
