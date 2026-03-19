# Stage1 (Unified VLM Training, No Task Heads)

This stage follows your figure's `stage1`:

- one VLM (`Qwen-VL`) is trained with a **single text-generation objective**
- target text contains structured labels for `box + pose + parsing`
- no separate detection/pose/parsing heads are used in this stage

## Data Format

Input JSONL each line:

```json
{
  "id": "000001",
  "image": "images/train/000001.jpg",
  "box": [{"bbox_xyxy": [100, 50, 300, 500]}],
  "pose": [{"keypoints": [[150, 100, 2], [160, 120, 2]]}],
  "parsing": [{"labels": ["face", "upper_clothes"], "segments": []}]
}
```

`target_text` can be provided directly; otherwise it will be auto-built.

## Build Stage1 JSONL

```bash
cd human/stage1
python tools/build_stage1_data.py \
  --input-jsonl data/human/unified/train.jsonl \
  --output-jsonl data/human/stage1/train.jsonl
```

## Train

```bash
cd human/stage1/scripts/train
./node.sh
```

Or run directly from project root:

```bash
python human/stage1/tools/train.py --config human/stage1/configs/qwen_vl_stage1.py
```

## Eval

```bash
cd human/stage1/scripts/test
./node.sh
```

## Outputs

- `stage1_*_predictions.jsonl`
- `stage1_*_metrics.json`

Metrics include:
- valid JSON rate
- box matching quality (IoU@0.5 based)
- pose keypoint distance
- parsing label-set F1

Per-epoch evaluation is enabled by `eval_each_epoch=True`.
`eval_max_samples_each_epoch=None` means evaluating on the full validation set.

## Runtime Prerequisites

- Python 3.10 or 3.11 (recommended for PyTorch ecosystem compatibility)
- `torch`, `transformers`, `accelerate`
- optional: `peft` if LoRA is enabled

---

## Five-Task Stage1 (box/pose/attribute/relation/parsing)

For your five-task setup, Stage1 uses:

- text CE loss for `box + pose + attribute + relation`
- text CE + parsing mask loss (`BCE + Dice`) for `parsing`
- `<SEG>` special token as parsing-mask routing token

### Stage0 Data Build

```bash
cd human/stage0
python tools/build_stage0_five_task_data.py \
  --input-jsonl data/human/unified/all.jsonl \
  --output-dir data/human/stage1_five_tasks
```

### Stage1 Five-Task Train

```bash
cd human/stage1
python tools/train_five_tasks.py \
  --config configs/qwen_vl_stage1_five_tasks.py \
  --train-jsonl data/human/stage1_five_tasks/train.jsonl \
  --val-jsonl data/human/stage1_five_tasks/val.jsonl \
  --image-root data/human \
  --mask-root data/human
```

Or:

```bash
cd human/stage1/scripts/train/five_tasks
./node.sh
```

### Stage1 Five-Task Eval

```bash
cd human/stage1
python tools/test_five_tasks.py \
  --config configs/qwen_vl_stage1_five_tasks.py \
  --checkpoint Outputs/stage1/qwen_vl_3b_five_tasks \
  --split val
```

Or:

```bash
cd human/stage1/scripts/test/five_tasks
./node.sh
```

### One-Command Full Pipeline

```bash
cd human
python stage1/tools/run_five_task_pipeline.py \
  --unified-jsonl data/human/unified/all.jsonl \
  --image-root data/human \
  --mask-root data/human
```
