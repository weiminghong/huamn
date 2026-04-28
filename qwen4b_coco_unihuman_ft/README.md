# Qwen COCO-UniHuman Box/Pose/Attribute Fine-Tuning

This directory is a small VLM fine-tuning and evaluation workspace for three tasks only:

- `box`
- `pose`
- `attribute`: `gender`, `age`

It intentionally does not train segmentation/parsing.

## Data Format

The converted JSONL records store normalized internal annotations, and SFT targets are emitted as tag-only text:

```xml
<answer>
<person>
<box>[120, 250, 480, 820]</box>
<pose>[250, 300, 2], ... exactly 17 keypoints ...</pose>
<gender>0</gender>
<age>25</age>
<score>1.0</score>
</person>
</answer>
```

Notes:

- Target text coordinates are normalized to `[0,1000]`.
- `bbox_xyxy` remains the evaluator's internal pixel-coordinate field after parsing.
- `keypoints` use COCO-17 order.
- `gender` follows COCO-UniHuman coding in this project: `0=male`, `1=female`.
- `person_id` only aligns the three text fields. Evaluation still matches people by IoU, not by output order.

## Prepare Data

```bash
bash qwen4b_coco_unihuman_ft/scripts/run_prepare_server_data.sh
```

If train and val images are in different folders:

```bash
TRAIN_IMAGE_ROOT=/home/priv/wmh_project/coco_person/images \
VAL_IMAGE_ROOT=/home/priv/wmh_project/coco_person/val/val2017 \
bash qwen4b_coco_unihuman_ft/scripts/run_prepare_server_data.sh
```

The default script already uses:

- train images: `/home/priv/wmh_project/coco_person/images`
- val images: `/home/priv/wmh_project/coco_person/val/val2017`

Your previous conversion produced `num_val=0`, which means the val image files referenced by `coco_unihuman_val_v1.json` were not found under the old configured `image_root`. Re-run the prepare script before val evaluation.

## Train

```bash
bash qwen4b_coco_unihuman_ft/scripts/run_train_swift_lora.sh
```

## Debug Inference With Visualization

This runs a random 10-image smoke test, saves predictions, computes scheme-2 metrics, and writes visualization images:

```bash
bash qwen4b_coco_unihuman_ft/scripts/run_eval_debug10_scheme2.sh
```

Equivalent direct command:

```bash
python qwen4b_coco_unihuman_ft/infer_eval_vis.py \
  --config qwen4b_coco_unihuman_ft/configs/eval_box_pose_attr_debug10_server.yaml
```

Outputs:

- `predictions.jsonl`: raw text and parsed structure
- `metrics.json`: scheme-2 metrics with per-image debug details
- `tag_parse_success_rate`: share of predictions parsed from `<answer>` tags
- `vis/`: GT and prediction visualization

Visualization colors:

- Green: GT
- Red: prediction
- Yellow keypoints: labeled but not visible (`v=1`)

## Full Val Evaluation

Run this after `val.jsonl` is non-empty:

```bash
bash qwen4b_coco_unihuman_ft/scripts/run_eval_val_scheme2.sh
```

Equivalent direct command:

```bash
python qwen4b_coco_unihuman_ft/infer_eval_vis.py \
  --config qwen4b_coco_unihuman_ft/configs/eval_box_pose_attr_val_server.yaml
```

## Evaluate Existing Predictions Only

If you already have a prediction JSONL:

```bash
python qwen4b_coco_unihuman_ft/eval_scheme2.py \
  --gt-jsonl /home/priv/wmh_project/coco_person/ft_box_pose_attr_data/val.jsonl \
  --pred-jsonl /home/priv/wmh_project/coco_person/ft_box_pose_attr_eval/val/predictions.jsonl \
  --output-json /home/priv/wmh_project/coco_person/ft_box_pose_attr_eval/val/metrics_recomputed.json
```

## Scheme-2 Evaluation Rule

The evaluator is designed for incomplete labels:

- Box is evaluated by IoU matching.
- Pose is evaluated only for GT people with valid keypoints.
- Attribute is evaluated only for matched GT people with valid `gender` or `age`.
- Predictions on people without pose/attribute labels do not become pose/attribute false positives.
- The main alignment is greedy one-to-one box matching by IoU, so multi-person output order does not need to match GT order.
