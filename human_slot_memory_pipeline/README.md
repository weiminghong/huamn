# Human Slot Memory Pipeline

This directory is an isolated research scaffold for the bidirectional human slot memory idea.
It does not modify or depend on the older training directories.

## Idea

The pipeline is organized around three trainable stages:

1. **Stage-1: Structured MLLM teacher**
   - Qwen-VL learns to decompose an image into person-level structured slots.
   - Each slot may contain box, pose, segmentation, attributes, action, relation, and caption fields.
   - Missing annotations are represented with task masks instead of negative labels.

2. **Stage-2: Dense human student**
   - DINOv3 features are decoded into dense person queries.
   - Person slots from the Stage-1 teacher supervise query semantics through slot alignment.
   - Box, pose, segmentation, attribute, and action heads are wired as separate modules.

3. **Stage-3: Dense-to-language memory loop**
   - Dense predictions are compressed into Human Memory Tokens.
   - Qwen-VL consumes those tokens for grounded caption, QA, and relation reasoning.
   - Consistency losses keep text outputs aligned with dense predictions.

## Directory

```text
human_slot_memory_pipeline/
  configs/                  # Server-ready YAML configs
  human_slot_memory/         # Python package
  scripts/                  # Shell wrappers for server use
  tools/                    # CLI entry points
```

## Server Paths

Default model paths keep weights outside this git repo:

```text
/home/priv/wmh_project/models/dinov3/vit_base_patch16_dinov3.lvd1689m
/home/priv/project/model/Qwen3.5-4B
```

## Quick Checks

Run a lightweight dry run without loading real Qwen/DINO weights:

```bash
cd /home/priv/wmh_project/human
PYTHONPATH=/home/priv/wmh_project/human python -m human_slot_memory_pipeline.tools.dry_run \
  --config human_slot_memory_pipeline/configs/pipeline_debug.yaml
```

All stage configs now point to raw COCO-UniHuman v1 annotations:

```text
/home/priv/wmh_project/coco_person/labels/coco_unihuman_train_v1.json
/home/priv/wmh_project/coco_person/labels/coco_unihuman_val_v1.json
```

Run the 20-image Stage-0 probe for `box + pose + gender + age`:

```bash
cd /home/priv/wmh_project/human
PYTHONPATH=/home/priv/wmh_project/human python -m human_slot_memory_pipeline.tools.eval_stage0_teacher \
  --config human_slot_memory_pipeline/configs/stage0_eval20.yaml
```

Prepare or launch Stage-1 teacher SFT:

```bash
cd /home/priv/wmh_project/human
PYTHONPATH=/home/priv/wmh_project/human python -m human_slot_memory_pipeline.tools.train_stage1_teacher \
  --config human_slot_memory_pipeline/configs/stage1_teacher.yaml
```

The current scaffold now covers:
- Stage-0 probe on raw v1 labels for `box`, `pose`, `gender`, `age`
- Stage-1 raw-v1 to SFT materialization plus legacy trainer handoff
- Stage-2 dense student with `box`, `pose`, `segment`, `attr(valid-aware)` heads
