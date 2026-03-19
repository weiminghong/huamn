# Stage0 (Five-Task Data Preparation)

Stage0 builds a unified five-task dataset for Stage1 multimodal SFT:

- `box`
- `pose`
- `attribute`
- `relation`
- `parsing` (as `<SEG>` token routing + mask files)

## Input JSONL

Each line can contain:

```json
{
  "id": "000001",
  "image": "images/train/000001.jpg",
  "box": [{"bbox_xyxy": [10, 20, 100, 200]}],
  "pose": [{"keypoints": [[20, 30, 2], [25, 40, 2]]}],
  "attribute": [{"gender": "female", "age_group": "adult"}],
  "relation": [{"subject": 0, "predicate": "riding", "object": "bike"}],
  "parsing": [{"label": "upper_clothes", "mask_path": "masks/000001_upper.png"}]
}
```

## Build Stage0 -> Stage1 Five-Task Files

```bash
cd human/stage0
python tools/build_stage0_five_task_data.py \
  --input-jsonl data/human/unified/all.jsonl \
  --output-dir data/human/stage1_five_tasks
```

Outputs:

- `train.jsonl`
- `val.jsonl`
- `test.jsonl`
- `summary.json`
