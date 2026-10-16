from datasets.coco_unihuman_dataset import CocoUnihumanDataset
import datasets  # noqa: F401

ann = '/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json'
img = '/share2/program/weiminghong/coco/val2017/'
try:
    ds = CocoUnihumanDataset(ann_file=ann, img_prefix=img, pipeline=[], test_mode=False)
except TypeError:
    ds = CocoUnihumanDataset(ann_file=ann, data_prefix={'img': img}, pipeline=[], test_mode=False)
print('len', len(ds))
item = ds[0]
print(type(item).__name__)
if isinstance(item, dict):
    print(sorted(item.keys())[:40])
    for key in ['img_info','ann_info','gt_bboxes','gt_labels','gt_attributes','gt_keypoints','gt_masks']:
        if key in item:
            value=item[key]
            shape=getattr(value,'shape',None)
            print(key, type(value).__name__, shape if shape is not None else '')
