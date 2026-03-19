_base_ = ["../_base_/default_runtime_pose.py"]

custom_imports = dict(
    imports=[
        "human.stage2.datasets",
        "mmpose.engine.optim_wrappers.layer_decay_optim_wrapper",
    ],
    allow_failed_imports=False,
)

model_name = "sapiens_1b"
embed_dim = 1536
num_layers = 40
pretrained_checkpoint = "../../../../pretrain/checkpoints/sapiens_1b/sapiens_1b_epoch_173_clean.pth"

data_root = "data/human"

evaluate_every_n_epochs = 1
vis_every_iters = 100
image_size = [768, 1024]  # width x height
sigma = 6
scale = 4
patch_size = 16
num_keypoints = 17
num_epochs = 210

train_cfg = dict(max_epochs=num_epochs, val_interval=evaluate_every_n_epochs)

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
    dict(type="LinearLR", begin=0, end=500, start_factor=0.001, by_epoch=False),
    dict(
        type="MultiStepLR",
        begin=0,
        end=num_epochs,
        milestones=[170, 200],
        gamma=0.1,
        by_epoch=True,
    ),
]

auto_scale_lr = dict(base_batch_size=512)

default_hooks = dict(
    checkpoint=dict(save_best="coco/AP", rule="greater", max_keep_ckpts=-1),
    visualization=dict(
        type="CustomPoseVisualizationHook", enable=True, interval=vis_every_iters, scale=scale
    ),
    logger=dict(type="LoggerHook", interval=10),
)

codec = dict(
    type="UDPHeatmap",
    input_size=(image_size[0], image_size[1]),
    heatmap_size=(int(image_size[0] / scale), int(image_size[1] / scale)),
    sigma=sigma,
)

model = dict(
    type="TopdownPoseEstimator",
    data_preprocessor=dict(
        type="PoseDataPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
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
        patch_cfg=dict(padding=2),
        init_cfg=dict(type="Pretrained", checkpoint=pretrained_checkpoint),
    ),
    head=dict(
        type="HeatmapHead",
        in_channels=embed_dim,
        out_channels=num_keypoints,
        deconv_out_channels=(768, 768),
        deconv_kernel_sizes=(4, 4),
        conv_out_channels=(768, 768),
        conv_kernel_sizes=(1, 1),
        loss=dict(type="KeypointMSELoss", use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(flip_test=True, flip_mode="heatmap", shift_heatmap=False),
)

train_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="RandomFlip", direction="horizontal"),
    dict(type="RandomHalfBody"),
    dict(type="RandomBBoxTransform"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PhotometricDistortion"),
    dict(
        type="Albumentation",
        transforms=[
            dict(type="Blur", p=0.1),
            dict(type="MedianBlur", p=0.1),
            dict(
                type="CoarseDropout",
                max_holes=1,
                max_height=0.4,
                max_width=0.4,
                min_holes=1,
                min_height=0.2,
                min_width=0.2,
                p=1.0,
            ),
        ],
    ),
    dict(type="GenerateTarget", encoder=codec),
    dict(type="PackPoseInputs"),
]

val_pipeline = [
    dict(type="LoadImage"),
    dict(type="GetBBoxCenterScale"),
    dict(type="TopdownAffine", input_size=codec["input_size"], use_udp=True),
    dict(type="PackPoseInputs"),
]

dataset_train = dict(
    type="HumanPoseDataset",
    data_root=data_root,
    data_mode="topdown",
    ann_file="pose/annotations/person_keypoints_train.json",
    data_prefix=dict(img="images/train/"),
)

train_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type="CombinedDataset",
        metainfo=dict(from_file="configs/_base_/datasets/coco.py"),
        datasets=[dataset_train],
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False, round_up=False),
    dataset=dict(
        type="HumanPoseDataset",
        data_root=data_root,
        data_mode="topdown",
        ann_file="pose/annotations/person_keypoints_val.json",
        data_prefix=dict(img="images/val/"),
        test_mode=True,
        pipeline=val_pipeline,
    ),
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/pose/annotations/person_keypoints_val.json",
)
test_evaluator = val_evaluator
