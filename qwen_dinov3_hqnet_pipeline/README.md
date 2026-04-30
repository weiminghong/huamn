# Qwen + DINOv3 + HumanQueryNet Pipeline

This folder is a clean Stage1/Stage2 pipeline that matches your sketch more closely than the older scattered `stage1/`, `stage2/`, and `qwen4b_coco_unihuman_ft/` directories.

It keeps two principles:

1. `Stage1` is a VLM instruction-tuning stage on structured human descriptions.
2. `Stage2` is a dense prediction stage with a discriminative backbone and task heads, following top-tier dense prediction practice rather than forcing all tasks through text generation.

## Why this design

Your drawing says:

- Stage1: fine-tune Qwen-VL on human-centric structured captioning.
- Stage2: freeze the Stage1 image encoder, connect to `DINOv3`, and train task-specific heads.

The direct path `Qwen visual tokens -> DINOv3 backbone` is not a natural engineering interface, because DINOv3 is itself a visual backbone and normally takes images, not another model's patch tokens.

So this folder implements the more defensible version:

- `Stage1`: Qwen-VL SFT for structured multi-person output.
- `Stage1 export`: save the Stage1 vision-tower weights and precompute per-image teacher dense features.
- `Stage2`: DINOv3 backbone + HumanQueryNet heads + offline Qwen teacher-feature distillation.

That keeps the Stage1 encoder frozen in Stage2, but avoids an awkward token-to-backbone cascade.

## References

The code here is intentionally aligned with these families of methods:

- `Qwen2.5-VL`: multimodal instruction tuning / generation backbone.
- `Deformable DETR`: dense set prediction with multi-scale deformable attention.
- `DINO` detector: stronger DETR training recipe with denoising and better matching.
- `DINOv3`: foundation vision backbone for dense visual features.
- `COCO-UniHuman / HumanQueryNet`: unified human box / pose / parsing / attribute heads and losses.

Useful official references:

- Qwen2.5-VL model docs: https://huggingface.co/docs/transformers/en/model_doc/qwen2_5_vl
- DINOv3 model docs: https://huggingface.co/docs/transformers/en/model_doc/dinov3
- DINO detector: https://openreview.net/forum?id=3mRwyG5one
- Deformable DETR: https://arxiv.org/abs/2010.04159
- Unified human parsing context: https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02688.pdf

## Folder layout

```text
qwen_dinov3_hqnet_pipeline/
  configs/
    stage1_qwen25vl_lora.yaml
    stage2_distill_hqnet_dinov3.py
  common/
    qwen_utils.py
  mmdet_ext/
    dinov3_backbone.py
    distill_human_query_net.py
    teacher_feature_pipeline.py
  tools/
    build_stage1_data.py
    train_stage1.py
    export_stage1_teacher_features.py
    train_stage2.py
```

## End-to-end flow

### 1. Build Stage1 SFT data

```bash
python qwen_dinov3_hqnet_pipeline/tools/build_stage1_data.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/images \
  --val-image-root /path/to/val2017 \
  --output-dir /path/to/ft_box_pose_attr_data
```

### 2. Train Stage1 Qwen-VL

```bash
python qwen_dinov3_hqnet_pipeline/tools/train_stage1.py \
  --model-name-or-path /path/to/Qwen2.5-VL-3B-Instruct \
  --train-jsonl /path/to/ft_box_pose_attr_data/train.jsonl \
  --val-jsonl /path/to/ft_box_pose_attr_data/val.jsonl \
  --image-root /path/to/images \
  --output-dir /path/to/outputs/stage1_qwen25vl_lora
```

### 3. Export Stage1 teacher features

```bash
python qwen_dinov3_hqnet_pipeline/tools/export_stage1_teacher_features.py \
  --model-name-or-path /path/to/Qwen2.5-VL-3B-Instruct \
  --adapter-path /path/to/outputs/stage1_qwen25vl_lora \
  --input-jsonl /path/to/ft_box_pose_attr_data/train.jsonl \
  --output-dir /path/to/stage1_teacher_cache/train
```

### 4. Train Stage2 DINOv3 + HumanQueryNet

```bash
python qwen_dinov3_hqnet_pipeline/tools/train_stage2.py \
  --config qwen_dinov3_hqnet_pipeline/configs/stage2_distill_hqnet_dinov3.py \
  --train-ann-json /path/to/coco_unihuman_train_v1.json \
  --val-ann-json /path/to/coco_unihuman_val_v1.json \
  --train-image-root /path/to/images \
  --val-image-root /path/to/val2017 \
  --train-teacher-index /path/to/stage1_teacher_cache/train/teacher_index.json \
  --val-teacher-index /path/to/stage1_teacher_cache/val/teacher_index.json \
  --dino-model-name facebook/dinov3-vits16-pretrain-lvd1689m \
  --work-dir /path/to/outputs/stage2_hqnet_dinov3
```

## What is intentionally not hidden

- Stage1 and Stage2 are not the same objective.
  Stage1 is completion-only CE.
  Stage2 is DETR/HQNet-style matching + task losses.
- Stage1 teacher distillation in Stage2 is offline by default.
  That is deliberate for throughput and engineering stability.
- If you later insist on online frozen-Qwen teacher distillation, the export utilities in `common/qwen_utils.py` are the starting point, but offline cache is the recommended default.
