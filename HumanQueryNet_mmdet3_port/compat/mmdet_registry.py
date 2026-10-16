from mmdet.registry import DATASETS, MODELS, TASK_UTILS, TRANSFORMS

HEADS = MODELS
LOSSES = MODELS
DETECTORS = MODELS
BACKBONES = MODELS
NECKS = MODELS
BBOX_ASSIGNERS = TASK_UTILS
MATCH_COST = TASK_UTILS
PIPELINES = TRANSFORMS


def build_loss(cfg):
    return MODELS.build(cfg)


def build_detector(cfg):
    return MODELS.build(cfg)


def build_head(cfg):
    return MODELS.build(cfg)


def build_assigner(cfg):
    return TASK_UTILS.build(cfg)


def build_match_cost(cfg):
    return TASK_UTILS.build(cfg)

TRANSFORMER = MODELS


def build_transformer(cfg):
    return MODELS.build(cfg)
