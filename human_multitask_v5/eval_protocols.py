from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence


@dataclass(frozen=True)
class EvalProtocol:
    dataset: str
    task: str
    scorer: str
    metrics: Sequence[str]
    area_policy: str
    comparable_group: str


PROTOCOLS = (
    EvalProtocol("coco_unihuman", "box", "coco_bbox", ("AP", "AP50", "AP75"), "no-small-person", "coco_v5"),
    EvalProtocol("coco_unihuman", "segment", "coco_segm", ("AP", "AP50", "AP75"), "no-small-person", "coco_v5"),
    EvalProtocol("coco_unihuman", "pose", "coco_keypoints", ("AP", "AP50", "AP75"), "no-small-person", "coco_v5"),
    EvalProtocol("coco_unihuman", "gender", "hqnet_attribute", ("AP",), "no-small-person", "coco_v5"),
    EvalProtocol("coco_unihuman", "age", "hqnet_attribute", ("AP",), "no-small-person", "coco_v5"),
    EvalProtocol("mhp_v2", "parsing", "mhp_official", ("mIoU", "APp50", "APpvol", "PCP50"), "official-all-persons", "mhp_v5"),
    EvalProtocol("mhp_v2", "pose", "mhp_pose16", ("PCKh",), "official-all-persons", "mhp_pose_v5"),
    EvalProtocol("psg", "relation", "openpsg_official", ("R@20", "R@50", "R@100", "mR@20", "mR@50", "mR@100"), "official", "psg_v5"),
    EvalProtocol("vaw", "attribute", "vaw_official", ("mAP", "mA", "F1"), "official", "vaw_v5"),
)


class EvaluationPlanError(ValueError):
    pass


def build_plan(
    dataset_tasks: Mapping[str, Iterable[str]]
) -> List[EvalProtocol]:
    requested = {
        (str(dataset), str(task))
        for dataset, tasks in dataset_tasks.items()
        for task in tasks
    }
    lookup = {(p.dataset, p.task): p for p in PROTOCOLS}
    missing = requested - set(lookup)
    if missing:
        raise EvaluationPlanError(
            f"no registered official protocol for {sorted(missing)}"
        )
    return [lookup[key] for key in sorted(requested)]


def assert_no_cross_protocol_average(rows: Sequence[Mapping]) -> None:
    """Reject a single AP made from incompatible scorers or units."""
    groups: Dict[str, set] = {}
    for row in rows:
        metric = str(row["metric"])
        group = str(row["comparable_group"])
        groups.setdefault(metric, set()).add(group)
    bad = {
        metric: value for metric, value in groups.items() if len(value) > 1
    }
    if bad:
        raise EvaluationPlanError(
            f"cross-protocol metric averaging is forbidden: {bad}"
        )
