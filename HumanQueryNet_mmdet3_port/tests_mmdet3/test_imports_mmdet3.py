import importlib

MODULES = [
    'models.dense_heads.human_query_head',
    'models.utils.uni_transformer',
    'models.utils.ms_pose_attn',
    'models.losses.focal_loss',
    'core.bbox.assigners.unihuman_hungarian_assigner',
    'core.bbox.match_costs.match_cost',
    'datasets.pipelines.uni_transforms',
]

for name in MODULES:
    importlib.import_module(name)
print('all HumanQueryNet core modules import under mmdet3')
