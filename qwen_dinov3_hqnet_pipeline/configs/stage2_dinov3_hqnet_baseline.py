import os

_base_ = ["./stage2_qwen_ie_dinov3_hqnet.py"]


def _env_bool(name, default=False):
    value = os.getenv(name)
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


DINOV3_MODEL = os.getenv(
    "DINOV3_MODEL_NAME",
    "/share2/program/DINOV3/DINOV3-safetensor/dinov3-vits16-pretrain-lvd1689m",
)
STAGE2_FREEZE_DINO = _env_bool("STAGE2_FREEZE_DINO", False)
WORK_DIR = os.getenv("STAGE2_WORK_DIR", "/share2/program/weiminghong/humanslot_outputs/stage2_dinov3_hqnet_baseline")

model = dict(
    backbone=dict(
        _delete_=True,
        type="DINOv3PyramidBackbone",
        model_name_or_path=DINOV3_MODEL,
        out_indices=(-4, -3, -2, -1),
        out_channels=256,
        num_outs=4,
        patch_size=16,
        freeze_backbone=STAGE2_FREEZE_DINO,
    )
)

evaluation = dict(interval=1, metric=["bbox", "attribute", "kpt", "segm"])
work_dir = WORK_DIR
