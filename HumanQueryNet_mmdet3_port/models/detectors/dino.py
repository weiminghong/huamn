# Copyright (c) OpenMMLab. All rights reserved.
from compat.mmdet_registry import DETECTORS
from mmdet.models.detectors import SingleStageDetector


@DETECTORS.register_module(force=True)
class DINO(SingleStageDetector):
    """HumanQueryNet's old DINO wrapper on MMDetection 3.x.

    The original code skipped MMDet2 DETR initialization and used only the
    single-stage detector backbone/neck/head plumbing. MMDet3 changed DETR into
    DetectionTransformer, so this port inherits SingleStageDetector directly.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
