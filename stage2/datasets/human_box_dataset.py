from mmdet.datasets import CocoDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class HumanBoxDataset(CocoDataset):
    """COCO-format person detection dataset."""

    METAINFO = dict(classes=("person",), palette=[(220, 20, 60)])
