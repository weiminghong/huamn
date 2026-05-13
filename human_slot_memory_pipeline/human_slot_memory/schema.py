from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import torch
except Exception:  # pragma: no cover - local lightweight import fallback
    class _TorchStub:
        Tensor = Any

    torch = _TorchStub()


COCO17_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


@dataclass
class TaskMask:
    box: bool = False
    pose: bool = False
    segmentation: bool = False
    parsing: bool = False
    attr: bool = False
    action: bool = False
    relation: bool = False
    caption: bool = False

    def as_dict(self) -> Dict[str, bool]:
        return {
            "box": self.box,
            "pose": self.pose,
            "segmentation": self.segmentation,
            "parsing": self.parsing,
            "attr": self.attr,
            "action": self.action,
            "relation": self.relation,
            "caption": self.caption,
        }


@dataclass
class PersonSlot:
    slot_id: int
    box_xyxy: Optional[List[float]] = None
    pose17_xyv: Optional[List[List[float]]] = None
    segmentation: Optional[Any] = None
    parsing: Optional[Any] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    action: Optional[str] = None
    relations: List[Dict[str, Any]] = field(default_factory=list)
    caption: Optional[str] = None
    task_mask: TaskMask = field(default_factory=TaskMask)
    teacher_embedding: Optional[torch.Tensor] = None

    def visible_pose_count(self) -> int:
        if not self.pose17_xyv:
            return 0
        return sum(1 for point in self.pose17_xyv if len(point) >= 3 and point[2] > 0)


@dataclass
class HumanSample:
    sample_id: str
    image_path: str
    width: int = 0
    height: int = 0
    slots: List[PersonSlot] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HumanBatch:
    samples: List[HumanSample]
    images: Optional[torch.Tensor] = None

    @property
    def batch_size(self) -> int:
        return len(self.samples)
