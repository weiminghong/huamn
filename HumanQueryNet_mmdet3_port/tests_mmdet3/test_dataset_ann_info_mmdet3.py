from datasets.coco_unihuman_dataset import CocoUnihumanDataset
import datasets  # noqa: F401

ann = '/share2/program/weiminghong/coco/unihuman_label/coco_unihuman_val_v1.json'
img = '/share2/program/weiminghong/coco/val2017/'
ds = CocoUnihumanDataset(ann_file=ann, img_prefix=img, pipeline=[], test_mode=False, indices=1)
print('len', len(ds))
info = ds.get_ann_info(0)
print(sorted(info.keys()))
for k,v in info.items():
    print(k, type(v).__name__, getattr(v, 'shape', ''))
