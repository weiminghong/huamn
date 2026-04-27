# Sapiens CIHP-Style Export

This folder contains a standalone exporter that runs the local `sapiens-main`
segmentation model on an image directory, uses the existing detection
`labels.json` to align person ids, and writes CIHP-style multi-person parsing
labels.

Default target paths are already set to the server layout mentioned in the task:

- checkpoint: `/share2/program/weiminghong/coco/checkpoints/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth`
- images: `/share2/program/weiminghong/coco/coco_person/images`
- detections: `/share2/program/weiminghong/coco/coco_person/labels_json/labels.json`
- output root: `/share2/program/weiminghong/coco/coco_person/parsing_cihp_style`

## Output layout

The script writes:

```text
parsing_cihp_style/
  Category_ids/
    000000000049.png
  Instance_ids/
    000000000049.png
  annotations.json
```

Notes:

- `Category_ids/*.png`: per-pixel CIHP parsing class ids.
- `Instance_ids/*.png`: per-pixel person ids aligned to detection `id`.
- `annotations.json`: index file with image paths and kept instance ids.
- Pixels outside all matched person boxes are set to background.
- If you also want raw CIHP semantic maps for debugging, pass `--save-npy-dir`.

## CIHP classes

The output category mask uses CIHP classes:

0. Background
1. Hat
2. Hair
3. Glove
4. Sunglasses
5. Upper-clothes
6. Dress
7. Coat
8. Socks
9. Pants
10. Jumpsuits
11. Scarf
12. Skirt
13. Face
14. Left-arm
15. Right-arm
16. Left-leg
17. Right-leg
18. Left-shoe
19. Right-shoe

## Current mapping source

Sapiens Goliath body-part predictions are mapped into CIHP classes before
instance assignment. Since Goliath does not natively predict some CIHP apparel
categories such as `Hat`, `Glove`, `Dress`, `Coat`, `Scarf`, or `Skirt`, those
classes will usually stay empty unless the generic clothing/body predictions are
later refined with another teacher.

## Example

```bash
python sapiens_seg_export/export_seg_json.py \
  --device cuda:0 \
  --batch-note "coco_person CIHP-style pseudo parsing labels"
```
