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

## Stage0 Box+Pose Eval (Qwen Structured JSON)

### 1) Build Eval JSONL From `person_with_instances.json` + `train/val.txt`

```bash
python stage0/tools/build_stage0_box_pose_eval_jsonl.py \
  --ann-json /share2/program/weiminghong/coco/coco_person/person_with_instances.json \
  --train-txt /share2/program/weiminghong/coco/coco_person/data_test/train.txt \
  --val-txt /share2/program/weiminghong/coco/coco_person/data_test/val.txt \
  --output-train-jsonl /share2/program/weiminghong/coco/coco_person/data_test/stage0_eval_train.jsonl \
  --output-val-jsonl /share2/program/weiminghong/coco/coco_person/data_test/stage0_eval_val.jsonl
```

Or use YAML config:

```bash
python stage0/tools/build_stage0_box_pose_eval_jsonl.py \
  --config stage0/configs/build_stage0_box_pose_eval.yaml
```

### 2) Run Box+Pose Evaluation On Val

```bash
python stage0/tools/eval_stage0_box_pose_qwenvl.py \
  --jsonl /share2/program/weiminghong/coco/coco_person/data_test/stage0_eval_val.jsonl \
  --image-root /share2/program/weiminghong/coco/coco_person/images \
  --model-path /share2/program/models/Qwen3.5-4B \
  --output-dir stage0/outputs/box_pose_eval_val \
  --eval-iou-thr 0.5 \
  --coco-ann-json /share2/program/weiminghong/coco/coco_person/person_with_instances.json \
  --coco-split-txt /share2/program/weiminghong/coco/coco_person/data_test/val.txt \
  --pose-pck-thr 0.2 \
  --batch-size 1
```

Or use YAML config (GPU 3):

```bash
python stage0/tools/eval_stage0_box_pose_qwenvl.py \
  --config stage0/configs/eval_stage0_box_pose_qwen35_val_gpu3.yaml
```

Main outputs:
- `predictions.jsonl`
- `metrics.json`
