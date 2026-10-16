from mmcv.cnn.bricks.transformer import MODELS as MMCV_TRANSFORMER_MODELS
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, TransformerLayerSequence

try:
    from mmdet.models.layers import (
        DeformableDetrTransformerDecoderLayer,
        DeformableDetrTransformerEncoderLayer,
    )
except Exception:  # pragma: no cover
    DeformableDetrTransformerDecoderLayer = None
    DeformableDetrTransformerEncoderLayer = None

# Old HumanQueryNet configs use DetrTransformerDecoderLayer with mmcv-style
# attn_cfgs/ffn_cfgs/operation_order. MMDetection 3.x's class with the same
# name has a different constructor, so keep the old semantic alias here.
MMCV_TRANSFORMER_MODELS.register_module(
    name='DetrTransformerDecoderLayer', module=BaseTransformerLayer, force=True)
MMCV_TRANSFORMER_MODELS.register_module(
    name='BaseTransformerLayer', module=BaseTransformerLayer, force=True)

for _cls in [
    DeformableDetrTransformerDecoderLayer,
    DeformableDetrTransformerEncoderLayer,
]:
    if _cls is not None:
        MMCV_TRANSFORMER_MODELS.register_module(module=_cls, force=True)

MMCV_TRANSFORMER_MODELS.register_module(name='DetrTransformerEncoder', module=TransformerLayerSequence, force=True)

from mmdet.models.layers import SinePositionalEncoding
MMCV_TRANSFORMER_MODELS.register_module(module=SinePositionalEncoding, force=True)
