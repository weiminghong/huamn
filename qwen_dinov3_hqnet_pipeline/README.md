# Qwen I_E -> DINOv3 -> HumanQueryNet Pipeline

This folder tracks Stage-1 and Stage-2 in the human-centric VLM+DINO design.

## Stage1

`Stage1` fine-tunes Qwen-VL on structured human-centric descriptions.

- Config: `configs/stage1_qwen25vl_lora.yaml`
- Launcher: `tools/train_stage1.py`

## Stage2

`Stage2` is the discriminative-head route in the paper diagram.

- Load the Stage1 Qwen checkpoint or adapter.
- Use only the Stage1 Qwen image encoder `I_E`.
- Freeze `I_E` by default inside the backbone wrapper.
- Feed image features through DINOv3.
- Use HumanQueryNet heads for box / pose / parsing / attribute prediction.
- Evaluate all available 2D COCO-UniHuman tasks with `metric=["bbox", "attribute", "kpt", "segm"]`.

`T_D` is not used in Stage2.

## Core files

- `configs/stage2_qwen_ie_dinov3_hqnet.py`: Qwen-IE + DINOv3 + HumanQueryNet config.
- `configs/stage2_dinov3_hqnet_baseline.py`: DINO-only baseline using the same HumanQueryNet heads, losses, and data.
- `mmdet_ext/dinov3_backbone.py`: DINOv3 wrapper and frozen-Qwen fusion backbone.
- `tools/train_stage2.py`: shared launcher for Qwen-fusion and DINO-only Stage2 runs.
- `tools/check_stage2_env.py`: preflight environment and path checker.
- `tests/test_stage2_static.py`: dependency-light static checks for Stage2 wiring.

## Fusion modes

The default Stage2 interaction is feature-level fusion rather than the earlier pixel-space bridge.

- `feature_add`: project the frozen Qwen feature map to each DINO pyramid level and add it to DINO features. This is the default paper-facing route.
- `feature_concat`: project the frozen Qwen feature map, concatenate with each DINO level, then reduce back to the DINO channel width.
- `input_add`: legacy bridge that converts Qwen features to a 3-channel perturbation and adds it to DINO input pixels.
- `input_replace`: legacy bridge that uses the 3-channel Qwen bridge as DINO input.
- `none`: load the Qwen wrapper but return plain DINO features; use only as a wiring ablation.

## Environment check

Run this before launching training. It does not load model weights.

```bash
export PYTHONPATH=/home/weiminghong/humanslot:/home/weiminghong/humanslot/COCO-UniHuman-main/HumanQueryNet:$PYTHONPATH
python qwen_dinov3_hqnet_pipeline/tools/check_stage2_env.py \
  --config qwen_dinov3_hqnet_pipeline/configs/stage2_qwen_ie_dinov3_hqnet.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/train2017_or_images \
  --val-image-root /path/to/val2017 \
  --qwen-model-name-or-path /path/to/Qwen2.5-VL-3B-Instruct \
  --dino-model-name /share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m
```

The expected runtime needs HumanQueryNet-compatible `mmcv`/`mmdet` plus `transformers`. A detector-only OpenMMLab 2.x/3.x environment is not enough if it misses the legacy `mmcv.utils.print_log` API used by the local HumanQueryNet code.

## DINO-only baseline

Run this first before claiming Qwen `I_E` helps. It uses DINOv3 and the same HumanQueryNet heads/losses/evaluation, without loading Qwen.

```bash
python qwen_dinov3_hqnet_pipeline/tools/train_stage2.py \
  --config qwen_dinov3_hqnet_pipeline/configs/stage2_dinov3_hqnet_baseline.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/train2017_or_images \
  --val-image-root /path/to/val2017 \
  --dino-model-name /share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m \
  --work-dir /share2/program/weiminghong/humanslot_outputs/stage2_dinov3_hqnet_baseline \
  --gpu-id 0 \
  --validate
```

## Qwen-IE feature-fusion run

```bash
python qwen_dinov3_hqnet_pipeline/tools/train_stage2.py \
  --config qwen_dinov3_hqnet_pipeline/configs/stage2_qwen_ie_dinov3_hqnet.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/train2017_or_images \
  --val-image-root /path/to/val2017 \
  --qwen-model-name-or-path /path/to/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter-path /path/to/stage1_qwen_adapter_or_empty \
  --dino-model-name /share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m \
  --fusion-mode feature_add \
  --work-dir /share2/program/weiminghong/humanslot_outputs/stage2_qwen_ie_dinov3_hqnet \
  --gpu-id 0 \
  --validate
```

Use `--fusion-mode feature_concat` as the main interaction ablation. Use `input_add` or `input_replace` only to compare against the previous minimal bridge design.

## Loss and evaluation contract

The Stage2 heads follow the HumanQueryNet/DETR-style set prediction route:

- Detection: Hungarian assignment with focal classification cost, L1 bbox cost, and GIoU cost; training losses are focal cls, L1 bbox, and GIoU.
- Pose: keypoint classification, keypoint L1, OKS, heatmap, and refinement losses.
- Parsing/segmentation: point-sampled sigmoid CE and Dice losses.
- Attribute: gender CE and age mean-variance softmax loss.
- Evaluation: `bbox`, `attribute`, `kpt`, and `segm` are enabled by default.

Paper-facing comparisons should report the DINO-only baseline and at least one feature-level Qwen-fusion run under the same dataset split, DINO checkpoint, image roots, max epochs, and evaluation metrics.
