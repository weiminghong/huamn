_base_ = ["../../../../det/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py"]

custom_imports = dict(imports=["human.stage2.datasets"], allow_failed_imports=False)

data_root = "data/human"
classes = ("person",)
metainfo = dict(classes=classes, palette=[(220, 20, 60)])

model = dict(bbox_head=dict(num_classes=1))

train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="HumanBoxDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="det/annotations/instances_train.json",
        data_prefix=dict(img="images/train/"),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
    ),
)

val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    dataset=dict(
        type="HumanBoxDataset",
        data_root=data_root,
        metainfo=metainfo,
        ann_file="det/annotations/instances_val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/det/annotations/instances_val.json",
    metric="bbox",
    classwise=True,
)
test_evaluator = val_evaluator

train_cfg = dict(max_epochs=100, val_interval=5)

default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=5,
        save_best="coco/bbox_mAP",
        rule="greater",
    )
)
