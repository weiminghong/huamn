import warnings

from compat.mmdet_registry import MODELS, BACKBONES, NECKS, HEADS, LOSSES, DETECTORS

ROI_EXTRACTORS = MODELS
SHARED_HEADS = MODELS


def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_roi_extractor(cfg):
    return ROI_EXTRACTORS.build(cfg)


def build_shared_head(cfg):
    return SHARED_HEADS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, please specify them in model',
            UserWarning)
    if train_cfg is not None or test_cfg is not None:
        cfg = cfg.copy()
        cfg.setdefault('train_cfg', train_cfg)
        cfg.setdefault('test_cfg', test_cfg)
    return DETECTORS.build(cfg)
