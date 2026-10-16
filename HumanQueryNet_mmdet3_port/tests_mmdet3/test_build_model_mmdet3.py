from mmengine.config import Config
from mmdet.registry import MODELS

import models  # noqa: F401 register local modules
import core  # noqa: F401 register assigners/costs

cfg = Config.fromfile('configs/human_query_net_all_task_r50_scratch.py')
cfg.model.bbox_head.with_smpl = False
cfg.model.bbox_head.pop('smpl_cfg', None)
model = MODELS.build(cfg.model)
print(type(model).__name__)
