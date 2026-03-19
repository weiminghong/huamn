from mmpose.datasets.datasets.body.coco_dataset import CocoDataset
from mmpose.registry import DATASETS


@DATASETS.register_module()
class HumanPoseDataset(CocoDataset):
    """COCO-format person keypoint dataset."""

    METAINFO = dict(from_file="configs/_base_/datasets/coco.py")
