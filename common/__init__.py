from .io_utils import dump_jsonl, load_jsonl
from .prompting import build_stage1_five_task_prompt, build_stage1_prompt
from .five_task_builder import build_five_task_target_text, normalize_five_task_sample
from .five_task_metrics import evaluate_five_task_records
from .stage1_metrics import evaluate_stage1_records
from .target_builder import build_stage1_target, parse_json_object_from_text

try:
    from .stage1_dataset import Stage1HumanDataset
except Exception:
    Stage1HumanDataset = None

try:
    from .stage1_five_task_dataset import Stage1FiveTaskDataset
except Exception:
    Stage1FiveTaskDataset = None

__all__ = [
    "load_jsonl",
    "dump_jsonl",
    "build_stage1_prompt",
    "build_stage1_five_task_prompt",
    "evaluate_stage1_records",
    "evaluate_five_task_records",
    "build_stage1_target",
    "build_five_task_target_text",
    "normalize_five_task_sample",
    "parse_json_object_from_text",
]

if Stage1HumanDataset is not None:
    __all__.append("Stage1HumanDataset")
if Stage1FiveTaskDataset is not None:
    __all__.append("Stage1FiveTaskDataset")
