#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=human_stage2
CONDA=/root/miniconda3/bin/conda
ENV=/root/miniconda3/envs/${ENV_NAME}
PY=${ENV}/bin/python
ROOT=/root/autodl-tmp
REPO=${ROOT}/humanslot
MODEL_DIR=${ROOT}/model/dinov3-vits16-pretrain-lvd1689m-timm
LOG=${REPO}/qwen_dinov3_hqnet_pipeline/tools/setup_stage2_env_a800_20260729.log

exec > >(tee -a "$LOG") 2>&1
echo "[$(date '+%F %T %Z')] setup start"

if [[ ! -x "$PY" ]]; then
  "$CONDA" create -y -n "$ENV_NAME" --clone /root/miniconda3/envs/human4b
fi

WHEEL_DIR=${ROOT}/wheels/stage2_torch241_cu121
mkdir -p "$WHEEL_DIR"
TORCH_WHEEL=${WHEEL_DIR}/torch-2.4.1+cu121-cp310-cp310-linux_x86_64.whl
TV_WHEEL=${WHEEL_DIR}/torchvision-0.19.1+cu121-cp310-cp310-linux_x86_64.whl
if [[ ! -s "$TORCH_WHEEL" || $(stat -c %s "$TORCH_WHEEL") -lt 798000000 ]]; then
  curl -L --fail --retry 8 -C - -o "$TORCH_WHEEL" 'https://mirrors.aliyun.com/pytorch-wheels/cu121/torch-2.4.1%2Bcu121-cp310-cp310-linux_x86_64.whl'
fi
if [[ ! -s "$TV_WHEEL" || $(stat -c %s "$TV_WHEEL") -lt 7000000 ]]; then
  curl -L --fail --retry 8 -C - -o "$TV_WHEEL" 'https://mirrors.aliyun.com/pytorch-wheels/cu121/torchvision-0.19.1%2Bcu121-cp310-cp310-linux_x86_64.whl'
fi
if ! "$PY" - <<'PY'
import sys, torch, torchvision
ok = torch.__version__.startswith("2.4.1") and torchvision.__version__.startswith("0.19.1")
print("torch stack present:", torch.__version__, torchvision.__version__, "ok=", ok)
sys.exit(0 if ok else 1)
PY
then
  "$PY" -m pip install --no-cache-dir --upgrade --force-reinstall "$TORCH_WHEEL" "$TV_WHEEL" -i https://pypi.tuna.tsinghua.edu.cn/simple
fi
"$PY" -m pip install --no-cache-dir --upgrade \
  'numpy==1.26.4' 'opencv-python==4.9.0.80' \
  'mmengine==0.10.7' 'mmdet==3.3.0' \
  'terminaltables>=3.1.0' 'prettytable>=3.10.0' 'shapely>=2.0.0' 'pytest>=8.0.0' 'timm>=1.0.20' 'transformers==5.4.0'
"$PY" -m pip install --no-cache-dir --upgrade \
  'mmcv==2.2.0' -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html

"$PY" - <<'PY'
from pathlib import Path
import sysconfig
p = Path(sysconfig.get_paths()["purelib"]) / "mmdet" / "__init__.py"
text = p.read_text(encoding='utf-8')
backup = p.with_suffix(p.suffix + '.orig_mmcv_lt22_20260729')
if not backup.exists():
    backup.write_text(text, encoding='utf-8')
old = "mmcv_maximum_version = '2.2.0'"
new = "mmcv_maximum_version = '2.3.0'"
if old in text:
    p.write_text(text.replace(old, new, 1), encoding='utf-8')
    print(f'patched {p}: mmcv maximum <2.3.0')
elif new in text:
    print(f'already patched {p}')
else:
    print(f'warning: expected mmdet version guard not found in {p}')
PY

mkdir -p "$MODEL_DIR"
export HF_HOME=${ROOT}/hf_stage2
export HUGGINGFACE_HUB_CACHE=${HF_HOME}/hub
export TRANSFORMERS_CACHE=${HF_HOME}/transformers
export HF_ENDPOINT=https://hf-mirror.com
"$PY" - <<PY
from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id='timm/vit_small_patch16_dinov3.lvd1689m',
    local_dir='${MODEL_DIR}',
    allow_patterns=['config.json', 'model.safetensors', 'README.md', 'LICENSE.md'],
)
print('DINO snapshot:', path)
PY

"$PY" - <<PY
from transformers import AutoModel
m = AutoModel.from_pretrained('${MODEL_DIR}', trust_remote_code=True)
print('DINO AutoModel:', type(m).__name__, 'parameters=', sum(p.numel() for p in m.parameters()))
del m
PY

cd "$REPO"
export PYTHONPATH="$REPO:$REPO/HumanQueryNet_mmdet3_port:${PYTHONPATH:-}"
"$PY" - <<'PY'
import torch, transformers, peft, mmcv, mmengine, mmdet
from mmcv.ops import MultiScaleDeformableAttention, point_sample
print({
    'torch': torch.__version__,
    'transformers': transformers.__version__,
    'peft': peft.__version__,
    'mmcv': mmcv.__version__,
    'mmengine': mmengine.__version__,
    'mmdet': mmdet.__version__,
    'cuda_available': torch.cuda.is_available(),
})
PY
"$PY" -m pytest -q qwen_dinov3_hqnet_pipeline/tests/test_stage2_static.py qwen_dinov3_hqnet_pipeline/tests/test_stage2_official_export.py
"$PY" HumanQueryNet_mmdet3_port/tests_mmdet3/test_imports_mmdet3.py

echo "[$(date '+%F %T %Z')] setup complete"
