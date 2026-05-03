# Qwen I_E -> DINOv3 -> HumanQueryNet Pipeline

This folder now tracks the two stages in your diagram directly.

## Stage1

`Stage1` fine-tunes Qwen-VL on structured human-centric descriptions.

- Config: `configs/stage1_qwen25vl_lora.yaml`
- Launcher: `tools/train_stage1.py`

## Stage2

`Stage2` uses only the Stage1 Qwen image encoder `I_E`.

- Load the Stage1 Qwen checkpoint or adapter.
- Extract Qwen visual features online during Stage2 training.
- Freeze `I_E`.
- Project the frozen Qwen feature map through a small bridge.
- Feed the bridged representation into DINOv3.
- Use HumanQueryNet heads for box / pose / parsing / attribute prediction.

The Stage2 route is online. There is no offline teacher-feature export step in this folder anymore.

## Core files

- `configs/stage2_qwen_ie_dinov3_hqnet.py`: Stage2 training config.
- `mmdet_ext/dinov3_backbone.py`: DINOv3 backbone wrapper and the online frozen-Qwen bridge backbone.
- `tools/train_stage2.py`: Stage2 launcher.

## Stage2 launch

```bash
python qwen_dinov3_hqnet_pipeline/tools/train_stage2.py \
  --config qwen_dinov3_hqnet_pipeline/configs/stage2_qwen_ie_dinov3_hqnet.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/images \
  --val-image-root /path/to/val2017 \
  --qwen-model-name-or-path /path/to/Qwen2.5-VL-3B-Instruct \
  --qwen-adapter-path /path/to/outputs/stage1_qwen25vl_lora \
  --dino-model-name facebook/dinov3-vits16-pretrain-lvd1689m \
  --work-dir /path/to/outputs/stage2_qwen_ie_dinov3_hqnet
```

## Notes

- `T_D` is not used in Stage2.
- The current bridge mode is an online Qwen-to-DINO feature injection route. It is the minimal serial interface that can run against the HF DINOv3 model API without reworking DINOv3 internals.
