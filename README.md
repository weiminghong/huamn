# Human Staged Pipeline

This folder is reorganized for staged development:

```text
human/
  common/      # shared utils across stage1/stage2/stage3
  stage0/      # five-task data build (box/pose/attribute/relation/parsing)
  stage1/      # VLM unified training (no task heads)
  stage2/      # head-based task training (box/pose/parsing)
  stage3/      # placeholder
```

## Why This Split

- `stage1`: single VLM objective (text generation for structured labels).
- `stage2`: discriminative heads for each task.
- `stage3`: reserved for later joint/distillation work.

`sapiens` itself is not a Qwen-VL text-generation stage. It is mainly
task-specific discriminative training (`mmdet/mmpose/mmseg` style).

## Shared Data Convention

```text
data/human/
  stage1/
    train.jsonl
    val.jsonl
    test.jsonl
  images/
    train/
    val/
  det/
    annotations/
  pose/
    annotations/
  parsing/
    images/
    annotations/
```

## Quick Entry

Stage1:

```bash
cd human/stage1/scripts/train && ./node.sh
cd human/stage1/scripts/test && ./node.sh
```

Stage0 + Stage1 (five tasks):

```bash
cd human
python stage1/tools/run_five_task_pipeline.py \
  --unified-jsonl data/human/unified/all.jsonl \
  --image-root data/human \
  --mask-root data/human
```

Stage2:

```bash
cd human/stage2/scripts/train/box && ./node.sh
cd human/stage2/scripts/train/pose && ./node.sh
cd human/stage2/scripts/train/parsing && ./node.sh
```

## Validated server source

See [retained implementations and validation scope](docs/VALIDATED_SERVER_SOURCE.md)
for the source selected from the server snapshot, its evidence limits, and the
preserved history. Dataset, checkpoint, and experiment-output assets remain on
the server.
