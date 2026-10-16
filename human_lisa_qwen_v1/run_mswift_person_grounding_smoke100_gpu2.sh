#!/usr/bin/env bash
set -euo pipefail
cd /home/weiminghong/humanslot
export CUDA_VISIBLE_DEVICES=2
export HF_HOME=/share2/program/weiminghong/hf_cache
export HUGGINGFACE_HUB_CACHE=/share2/program/weiminghong/hf_cache/hub
export TRANSFORMERS_CACHE=/share2/program/weiminghong/hf_cache/transformers
/home/weiminghong/yes/envs/human_v2/bin/swift sft /home/weiminghong/humanslot/human_lisa_qwen_v1/mswift_person_grounding_smoke100.yaml 2>&1 | tee /share2/program/weiminghong/humanslot_outputs/mswift_person_grounding_v1_20260714_retry1/sft_smoke100_launch.log
