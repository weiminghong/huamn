#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
LOG=/share2/program/weiminghong/humanslot_outputs/mswift_person_grounding_v1_20260714_retry1/sft_full_vlora_3epoch_launch.log
/home/weiminghong/yes/envs/human_v2/bin/swift sft /home/weiminghong/humanslot/human_lisa_qwen_v1/mswift_person_grounding_full_vlora_3epoch_gpu2.yaml 2>&1 | tee "$LOG"
