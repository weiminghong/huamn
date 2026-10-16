# HumanQueryNet MMDetection 3 Port Status

## Environment
- Conda env: `/home/weiminghong/yes/envs/human_full`
- Torch: `2.4.1+cu121`
- Transformers: `5.4.0`
- MMEngine: `0.10.7`
- MMDetection: `3.3.0`
- MMCV: `2.2.0` from OpenMMLab `cu121/torch2.4.0` wheel
- Numpy constrained to `<2` (`1.26.4` observed)
- OpenCV constrained to `opencv-python==4.9.0.80`

## Local env patch
`mmdet==3.3.0` rejects `mmcv==2.2.0` via version check, although this is the available prebuilt wheel for torch2.4/cu121. The env file below was backed up and then patched to allow `<2.3.0`:

- `/home/weiminghong/yes/envs/human_full/lib/python3.10/site-packages/mmdet/__init__.py`
- backup: `/home/weiminghong/yes/envs/human_full/lib/python3.10/site-packages/mmdet/__init__.py.orig_mmcv_lt22_20260709`

## Port copy
- Port path: `/home/weiminghong/humanslot/HumanQueryNet_mmdet3_port`
- Original path left unchanged: `/home/weiminghong/humanslot/COCO-UniHuman-main/HumanQueryNet`

## Smoke tests
Run from the port path with `human_full`:

```bash
PYTHONPATH="$PWD:$PYTHONPATH" python tests_mmdet3/test_imports_mmdet3.py
PYTHONPATH="$PWD:$PYTHONPATH" python tests_mmdet3/test_build_model_mmdet3.py
PYTHONPATH="$PWD:$PYTHONPATH" python tests_mmdet3/test_extract_feat_mmdet3.py
PYTHONPATH="$PWD:$PYTHONPATH" python tests_mmdet3/test_forward_dummy_mmdet3.py
```

Current smoke status:
- Core imports: pass
- Model build: pass with `with_smpl=False`
- `extract_feat` on random image: pass
- `forward_dummy` on random image: pass

## Known remaining work
- Full train/loss path under MMEngine runner is not verified yet.
- Dataset pipeline was import-ported, but dataloader and annotation conversion still need a real COCO-UniHuman sample test.
- SMPL path is disabled in smoke because the prior file path was missing from the port runtime.
- `force_fp32`, `DataContainer`, `collate`, and `mmcv.jit` are compatibility shims; training semantics need targeted tests before paper-facing experiments.
- The restored local `DeformableDetrTransformerDecoder` supports old config construction; full numerical parity with the old mmdet2 implementation is not claimed yet.
