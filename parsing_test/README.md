## Qwen-VL + SAM2 Parsing Baseline

This folder implements a baseline parsing evaluation pipeline:

1. Qwen-VL reads an input image and outputs structured parsing text.
2. The structured text contains part labels and bounding boxes.
3. SAM2 takes each predicted part box and generates a mask.
4. All part masks are merged into a semantic parsing map.
5. The prediction is compared with LIP-style ground-truth masks and mIoU is reported.

### Expected Qwen Output Schema

The prompt asks Qwen-VL to return one JSON object:

```json
{
  "persons": [
    {
      "person_bbox": [x1, y1, x2, y2],
      "parts": [
        {"label": "hat", "bbox": [x1, y1, x2, y2]},
        {"label": "hair", "bbox": [x1, y1, x2, y2]},
        {"label": "upper_clothes", "bbox": [x1, y1, x2, y2]}
      ]
    }
  ]
}
```

### LIP Layout Assumption

This code assumes a split with both images and parsing annotations available:

```text
LIP/
  val_images/
  val_segmentations/
```

Or:

```text
LIP/
  test_images/
  test_segmentations/
```

If your local naming differs, pass the paths explicitly by CLI arguments.

### Example

```bash
python parsing_test/evaluate_lip.py ^
  --image-dir D:\datasets\LIP\val_images ^
  --ann-dir D:\datasets\LIP\val_segmentations ^
  --output-dir Outputs/parsing_test/lip_val ^
  --max-samples 100
```

### Notes

- This is a baseline for evaluation and pipeline validation, not a strong human parsing method.
- The quality depends heavily on Qwen predicting usable part boxes.
- Official LIP test sets without public labels cannot be evaluated locally for mIoU.
