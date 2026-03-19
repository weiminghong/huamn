from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class HumanParsingDataset(BaseSegDataset):
    """Directory-based human parsing dataset.

    Expected by default:
      data_prefix.img_path: parsing/images/{split}
      data_prefix.seg_map_path: parsing/annotations/{split}
    """

    METAINFO = dict(
        classes=(
            "background",
            "hat",
            "hair",
            "glove",
            "sunglasses",
            "upper_clothes",
            "dress",
            "coat",
            "socks",
            "pants",
            "jumpsuits",
            "scarf",
            "skirt",
            "face",
            "left_arm",
            "right_arm",
            "left_leg",
            "right_leg",
            "left_shoe",
            "right_shoe",
        ),
        palette=[
            [0, 0, 0],
            [128, 0, 0],
            [255, 0, 0],
            [0, 85, 0],
            [170, 0, 51],
            [255, 85, 0],
            [0, 0, 85],
            [0, 119, 221],
            [85, 85, 0],
            [0, 85, 85],
            [85, 51, 0],
            [52, 86, 128],
            [0, 128, 0],
            [0, 0, 255],
            [51, 170, 221],
            [0, 255, 255],
            [85, 255, 170],
            [170, 255, 85],
            [255, 255, 0],
            [255, 170, 0],
        ],
    )

    def __init__(self, **kwargs):
        super().__init__(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)
