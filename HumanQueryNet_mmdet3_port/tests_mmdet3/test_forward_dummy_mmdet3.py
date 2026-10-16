import torch
from mmengine.config import Config
from mmdet.registry import MODELS

import models  # noqa: F401
import core  # noqa: F401

cfg = Config.fromfile('configs/human_query_net_all_task_r50_scratch.py')
cfg.model.bbox_head.with_smpl = False
cfg.model.bbox_head.pop('smpl_cfg', None)
cfg.model.backbone.init_cfg = None
model = MODELS.build(cfg.model)
model.eval()
with torch.no_grad():
    out = model.forward_dummy(torch.randn(1, 3, 128, 128))
print(type(out).__name__)
