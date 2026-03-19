_base_ = ["../_base_/default_runtime_seg.py"]

custom_imports = dict(
    imports=["human.stage2.datasets", "mmseg.engine.optimizers.layer_decay_optim_wrapper"],
    allow_failed_imports=False,
)

model_name = "sapiens_1b"
embed_dim = 1536
num_layers = 40
patch_size = 16
num_epochs = 200
num_classes = 20

data_root = "data/human"
pretrained_checkpoint = "../../../../pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth"

image_size = (768, 1024)  # width, height
evaluate_every_n_epochs = 1
vis_every_iters = 100

norm_cfg = dict(type="SyncBN", requires_grad=True)
data_preprocessor = dict(
    type="SegDataPreProcessor",
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    size=(image_size[1], image_size[0]),
    seg_pad_val=255,
)

model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type="mmpretrain.VisionTransformer",
        arch=model_name,
        img_size=(image_size[1], image_size[0]),
        patch_size=patch_size,
        qkv_bias=True,
        final_norm=True,
        drop_path_rate=0.0,
        with_cls_token=False,
        out_type="featmap",
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_checkpoint),
    ),
    decode_head=dict(
        type="VitHead",
        in_channels=embed_dim,
        channels=768,
        deconv_out_channels=(768, 768, 768),
        deconv_kernel_sizes=(4, 4, 4),
        conv_out_channels=(768, 768, 768),
        conv_kernel_sizes=(1, 1, 1),
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    train_cfg=dict(),
    test_cfg=dict(mode="whole"),
)

optim_wrapper = dict(
    optimizer=dict(type="AdamW", lr=5e-4, betas=(0.9, 0.999), weight_decay=0.1),
    paramwise_cfg=dict(
        num_layers=num_layers,
        layer_decay_rate=0.85,
        custom_keys={
            "bias": dict(decay_multi=0.0),
            "pos_embed": dict(decay_mult=0.0),
            "relative_position_bias_table": dict(decay_mult=0.0),
            "norm": dict(decay_mult=0.0),
        },
    ),
    constructor="LayerDecayOptimWrapperConstructor",
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

param_scheduler = [
    dict(type="LinearLR", start_factor=1e-3, begin=0, end=400, by_epoch=False),
    dict(
        type="PolyLR",
        eta_min=0.0,
        power=1.0,
        begin=0,
        end=num_epochs,
        by_epoch=True,
    ),
]

train_cfg = dict(
    type="EpochBasedTrainLoop",
    max_epochs=num_epochs,
    val_interval=evaluate_every_n_epochs,
)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")
log_processor = dict(by_epoch=True)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=10),
    param_scheduler=dict(type="ParamSchedulerHook"),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_best="mIoU",
        rule="greater",
    ),
    visualization=dict(
        type="GeneralSegVisualizationHook",
        interval=vis_every_iters,
        max_samples=4,
        vis_image_width=384,
        vis_image_height=512,
    ),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="LoadAnnotations"),
    dict(type="RandomResize", scale=(768, 1024), ratio_range=(0.2, 2.0), keep_ratio=True),
    dict(type="RandomCrop", crop_size=(1024, 768), cat_max_ratio=0.75),
    dict(type="Resize", scale=(768, 1024), keep_ratio=False),
    dict(type="RandomRotate", prob=0.5, degree=45, seg_pad_val=0),
    dict(
        type="RandomFlip",
        prob=0.5,
        swap_seg_labels=[(14, 15), (16, 17), (18, 19)],
    ),
    dict(type="PhotoMetricDistortion"),
    dict(type="PackSegInputs"),
]

test_pipeline = [
    dict(type="LoadImage"),
    dict(type="Resize", scale=(768, 1024), keep_ratio=False),
    dict(type="LoadAnnotations"),
    dict(type="PackSegInputs"),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="HumanParsingDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="parsing/images/train",
            seg_map_path="parsing/annotations/train",
        ),
        reduce_zero_label=False,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=2,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type="HumanParsingDataset",
        data_root=data_root,
        data_prefix=dict(
            img_path="parsing/images/val",
            seg_map_path="parsing/annotations/val",
        ),
        reduce_zero_label=False,
        test_mode=True,
        pipeline=test_pipeline,
    ),
)

test_dataloader = val_dataloader
val_evaluator = dict(type="IoUMetric", iou_metrics=["mIoU"])
test_evaluator = val_evaluator
