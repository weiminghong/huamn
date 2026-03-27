## Box Task: Qwen-VL Structured Generation

This folder contains a box-only pipeline for the structured-text baseline:

1. Dataset loader:
   - image file
   - paired txt label file with the same stem
   - each txt line is `x y w h`
2. Prompt and target-text building for Qwen-VL.
3. Validation:
   - parse structured JSON from model output
   - convert predicted boxes to xyxy
   - score-threshold + NMS
   - compare with GT using IoU-based metrics

### Expected Dataset Layout

```text
data_root/
  images/
    train/
    val/
  labels/
    train/
    val/
```

For each image:

```text
images/train/000001.jpg
labels/train/000001.txt
```

Label file format:

```text
x y w h
x y w h
```

### Config

See [box_qwenvl_data.py](/D:/科研项目/人体感知/human/box_task/configs/box_qwenvl_data.py).

### Validation Example

```powershell
python box_task/val.py `
  --config box_task/configs/box_qwenvl_data.py `
  --checkpoint Qwen/Qwen2.5-VL-3B-Instruct `
  --split val `
  --output-dir Outputs/box_task/val
```

### Expected Qwen Output Schema

```json
{
  "box": [
    {"bbox_xyxy": [100, 50, 300, 500], "score": 0.98},
    {"bbox_xyxy": [420, 80, 700, 900], "score": 0.91}
  ]
}
```

`score` is optional. If absent, validation uses `1.0`.
