import os

_base_ = [
    "../../COCO-UniHuman-main/HumanQueryNet/configs/default_runtime.py",
]

custom_imports = dict(
    imports=[
        "qwen_dinov3_hqnet_pipeline.mmdet_ext.dinov3_backbone",
        "datasets.coco_unihuman_dataset",
        "datasets.pipelines.uni_transforms",
    ],
    allow_failed_imports=False,
)


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name, default):
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return int(default)
    return int(value)


DATA_ROOT = os.getenv("COCO_UNIHUMAN_ROOT", "/path/to/coco_person")
TRAIN_ANN = os.getenv("COCO_UNIHUMAN_TRAIN_JSON", os.path.join(DATA_ROOT, "labels", "coco_unihuman_train_v1.json"))
VAL_ANN = os.getenv("COCO_UNIHUMAN_VAL_JSON", os.path.join(DATA_ROOT, "labels", "coco_unihuman_val_v1.json"))
TRAIN_IMG_ROOT = os.getenv("COCO_UNIHUMAN_TRAIN_IMG_ROOT", os.path.join(DATA_ROOT, "images"))
VAL_IMG_ROOT = os.getenv("COCO_UNIHUMAN_VAL_IMG_ROOT", os.path.join(DATA_ROOT, "val", "val2017"))
QWEN_STAGE1_MODEL = os.getenv("STAGE1_QWEN_MODEL_NAME_OR_PATH", "/path/to/Qwen2.5-VL-3B-Instruct")
QWEN_STAGE1_ADAPTER = os.getenv("STAGE1_QWEN_ADAPTER_PATH", "")
DINOV3_MODEL = os.getenv(
    "DINOV3_MODEL_NAME",
    "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m",
)
WORK_DIR = os.getenv("STAGE2_WORK_DIR", os.path.join(DATA_ROOT, "outputs", "stage2_qwen_ie_dinov3_hqnet"))
STAGE2_FUSION_MODE = os.getenv("STAGE2_FUSION_MODE", "feature_add")
STAGE2_FREEZE_DINO = _env_bool("STAGE2_FREEZE_DINO", False)
STAGE2_MAX_EPOCHS = _env_int("STAGE2_MAX_EPOCHS", 100)

# COCO-UniHuman supports these 2D metrics in the local HumanQueryNet dataset wrapper.
STAGE2_EVAL_METRICS = ["bbox", "attribute", "kpt", "segm"]

dataset_type = "CocoUnihumanDataset"
img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type="LoadImageFromFile", to_float32=True),
    dict(type="LoadMultitaskInstanceAnnotations", with_bbox=True, with_mask=True, extra_anno_list=["attributes", "keypoints", "areas", "smpl_pose", "smpl_betas", "valids"]),
    dict(type="KPRandomFlip", flip_ratio=0.5),
    dict(
        type="AutoAugment",
        policies=[
            [dict(type="KPResize", img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], multiscale_mode="value", keep_ratio=True)],
            [
                dict(type="KPResize", img_scale=[(400, 4200), (500, 4200), (600, 4200)], multiscale_mode="value", keep_ratio=True),
                dict(type="UniRandomCrop", crop_type="absolute_range", crop_size=(384, 600), crop_list=["gt_keypoints", "gt_attributes", "gt_areas", "gt_labels", "gt_masks"], allow_negative_crop=True),
                dict(type="KPResize", img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), (608, 1333), (640, 1333), (672, 1333), (704, 1333), (736, 1333), (768, 1333), (800, 1333)], multiscale_mode="value", override=True, keep_ratio=True),
            ],
        ],
    ),
    dict(type="PhotoMetricDistortion", brightness_delta=32, contrast_range=(0.5, 1.5), saturation_range=(0.5, 1.5), hue_delta=18),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="Pad", size_divisor=1),
    dict(type="MultiTaskFormatBundle", keys=["proposals", "gt_bboxes", "gt_bboxes_ignore", "gt_labels", "gt_attributes", "gt_keypoints", "gt_areas", "gt_smpl_pose", "gt_smpl_betas", "gt_valids"]),
    dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels", "gt_attributes", "gt_keypoints", "gt_areas", "gt_masks", "gt_smpl_pose", "gt_smpl_betas", "gt_valids"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=1),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]

model = dict(
    type="HumanQueryNet",
    backbone=dict(
        type="FrozenQwenDINOv3Backbone",
        qwen_model_name_or_path=QWEN_STAGE1_MODEL,
        qwen_adapter_path=QWEN_STAGE1_ADAPTER,
        qwen_torch_dtype="bfloat16",
        qwen_merge_lora=True,
        qwen_input_mean=img_norm_cfg["mean"],
        qwen_input_std=img_norm_cfg["std"],
        model_name_or_path=DINOV3_MODEL,
        out_indices=(-4, -3, -2, -1),
        out_channels=256,
        num_outs=4,
        patch_size=16,
        freeze_backbone=STAGE2_FREEZE_DINO,
        fusion_mode=STAGE2_FUSION_MODE,
        bridge_hidden_channels=64,
        bridge_scale=1.0,
    ),
    neck=dict(type="ChannelMapper", in_channels=[256, 256, 256, 256], kernel_size=1, out_channels=256, act_cfg=None, norm_cfg=dict(type="GN", num_groups=32), num_outs=5),
    bbox_head=dict(
        type="HumanQueryHead",
        num_query=300,
        num_classes=1,
        with_kp=True,
        with_det=True,
        with_seg=True,
        with_attr=True,
        with_smpl=False,
        attr_cfg=[dict(name="gender", num_classes=1), dict(name="age", num_classes=85)],
        transformer=dict(
            type="UniTransformer",
            strides=[8, 16, 32, 64],
            in_channels=[256, 256, 256, 256],
            feat_channels=256,
            out_channels=256,
            num_outs=3,
            norm_cfg=dict(type="GN", num_groups=32),
            act_cfg=dict(type="ReLU"),
            num_keypoints=17,
            with_kpt_refine=True,
            encoder=dict(type="UnihumanTransformerEncoder", num_layers=6, transformerlayers=dict(type="BaseTransformerLayer", attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, num_levels=4, dropout=0.0), ffn_cfgs=dict(type="FFN", feedforward_channels=2048, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)), operation_order=("self_attn", "norm", "ffn", "norm"))),
            decoder=dict(type="UniTransformerDecoder", num_layers=6, num_feat=128, return_intermediate=True, transformerlayers=dict(type="DetrTransformerDecoderLayer", feedforward_channels=2048, attn_cfgs=[dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.0), dict(type="MultiScaleDeformableAttention", embed_dims=256, num_levels=4, dropout=0.0)], ffn_cfgs=dict(type="FFN", feedforward_channels=2048, num_fcs=2, ffn_drop=0.0, act_cfg=dict(type="ReLU", inplace=True)), operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"))),
            kpt_refine_decoder=dict(type="DeformableDetrTransformerDecoder", num_layers=2, return_intermediate=True, transformerlayers=dict(type="DetrTransformerDecoderLayer", attn_cfgs=[dict(type="MultiheadAttention", embed_dims=256, num_heads=8, dropout=0.1), dict(type="MultiScaleDeformableAttention", embed_dims=256, im2col_step=128)], feedforward_channels=1024, ffn_dropout=0.1, operation_order=("self_attn", "norm", "cross_attn", "norm", "ffn", "norm"))),
            hm_encoder=dict(type="DetrTransformerEncoder", num_layers=1, transformerlayers=dict(type="BaseTransformerLayer", attn_cfgs=dict(type="MultiScaleDeformableAttention", embed_dims=256, num_levels=1), feedforward_channels=1024, ffn_dropout=0.1, operation_order=("self_attn", "norm", "ffn", "norm"))),
        ),
        kp_cfg=dict(with_kpt_refine=True, num_kpt_fcs=2, num_keypoints=17, with_hm=True, as_two_stage=True, loss_kpt_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0), loss_kpt=dict(type="L1Loss", loss_weight=50.0), loss_kpt_rpn=dict(type="L1Loss", loss_weight=50.0), loss_oks=dict(type="OKSLoss", loss_weight=1.5), loss_hm=dict(type="CenterFocalLoss", loss_weight=4.0), loss_kpt_refine=dict(type="L1Loss", loss_weight=60.0), loss_oks_refine=dict(type="OKSLoss", loss_weight=2.0)),
        seg_cfg=dict(feat_channels=256, out_channels=256, loss_mask=dict(type="CrossEntropyLoss", use_sigmoid=True, reduction="mean", loss_weight=8.0), loss_dice=dict(type="DiceLoss", use_sigmoid=True, activate=True, reduction="mean", naive_dice=True, eps=1.0, loss_weight=5.0), train_cfg=dict(num_points=12544, oversample_ratio=3.0, importance_sample_ratio=0.75, mask_downsample_ratio=4, sampler=dict(type="MaskPseudoSampler"))),
        num_feature_levels=4,
        in_channels=256,
        sync_cls_avg_factor=True,
        as_two_stage=True,
        with_box_refine=True,
        dn_cfg=dict(type="CdnQueryGenerator", noise_scale=dict(label=0.5, box=1.0), group_cfg=dict(dynamic=True, num_groups=None, num_dn_queries=100)),
        positional_encoding=dict(type="SinePositionalEncoding", num_feats=128, temperature=20, normalize=True),
        loss_cls=dict(type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),
        loss_attrs=[dict(type="CrossEntropyLoss", use_sigmoid=True, loss_weight=1.0), dict(type="MeanVarianceSoftmaxLoss", weight_mean=0.02, weight_var=0.1, loss_weight=0.1)],
    ),
    train_cfg=dict(uni_assigner=dict(type="UnihumanHungarianAssigner", cls_cost=dict(type="FocalLossCost", weight=1.0), reg_cost=dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"), iou_cost=dict(type="IoUCost", iou_mode="giou", weight=2.0), kpt_cost=dict(type="KptL1Cost", weight=50.0), oks_cost=dict(type="OksV4Cost", weight=1.5), mask_cost=dict(type="CrossEntropyLossCost", weight=8.0, use_sigmoid=True), dice_cost=dict(type="DiceCost", weight=5.0, pred_act=True, eps=1.0), kpt_cls_cost=dict(type="FocalLossCostV2", weight=1.0))),
    test_cfg=dict(max_per_img=100),
)

optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.0001, paramwise_cfg=dict(custom_keys={"backbone": dict(lr_mult=0.1)}))
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
lr_config = dict(policy="step", step=[80])
runner = dict(type="EpochBasedRunner", max_epochs=STAGE2_MAX_EPOCHS)
log_config = dict(interval=50, hooks=[dict(type="TextLoggerHook")])
checkpoint_config = dict(interval=2)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(type=dataset_type, ann_file=TRAIN_ANN, img_prefix=TRAIN_IMG_ROOT, pipeline=train_pipeline),
    val=dict(type=dataset_type, ann_file=VAL_ANN, img_prefix=VAL_IMG_ROOT, pipeline=test_pipeline),
    test=dict(type=dataset_type, ann_file=VAL_ANN, img_prefix=VAL_IMG_ROOT, pipeline=test_pipeline),
)
evaluation = dict(interval=1, metric=["bbox", "attribute", "kpt", "segm"])
auto_scale_lr = dict(base_batch_size=16)
find_unused_parameters = True
work_dir = WORK_DIR
