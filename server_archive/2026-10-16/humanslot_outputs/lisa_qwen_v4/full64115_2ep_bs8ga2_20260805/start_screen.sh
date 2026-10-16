#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/humanslot
screen -dmS v4_full_stage1_20260805 bash /root/autodl-tmp/humanslot_outputs/lisa_qwen_v4/full64115_2ep_bs8ga2_20260805/launch.sh
