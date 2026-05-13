from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from torch.utils.data import Dataset

from ..schema import HumanBatch, HumanSample, PersonSlot, TaskMask


def _read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _has_pose(pose: Any) -> bool:
    return isinstance(pose, list) and len(pose) == 17 and any(len(p) >= 3 and p[2] > 0 for p in pose)


def _entry_person_id(entry: Any, fallback: int) -> int:
    if isinstance(entry, dict) and "person_id" in entry:
        try:
            return int(entry["person_id"])
        except (TypeError, ValueError):
            return fallback
    return fallback


def _normalize_box(entry: Any) -> Optional[List[float]]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        coords = entry.get("bbox_xyxy") or entry.get("box") or entry.get("bbox")
    else:
        coords = entry
    if not isinstance(coords, list) or len(coords) != 4:
        return None
    return [float(v) for v in coords]


def _normalize_pose(entry: Any) -> Optional[List[List[float]]]:
    if entry is None:
        return None
    if isinstance(entry, dict):
        keypoints = entry.get("keypoints") or entry.get("pose")
    else:
        keypoints = entry
    if not _has_pose(keypoints):
        return None
    return [[float(value) for value in point] for point in keypoints]


def _normalize_attr(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, dict):
        return {k: v for k, v in entry.items() if k != "person_id"}
    return {}


def _build_slots(record: Dict[str, Any]) -> List[PersonSlot]:
    boxes = record.get("box", []) or record.get("bbox_xyxy", []) or []
    poses = record.get("pose", []) or []
    attrs = record.get("attribute", []) or []
    pose_by_person = {_entry_person_id(entry, idx + 1): _normalize_pose(entry) for idx, entry in enumerate(poses)}
    attr_by_person = {_entry_person_id(entry, idx + 1): _normalize_attr(entry) for idx, entry in enumerate(attrs)}
    slots: List[PersonSlot] = []
    for index, box in enumerate(boxes):
        person_id = _entry_person_id(box, index + 1)
        norm_box = _normalize_box(box)
        pose = pose_by_person.get(person_id)
        attr = attr_by_person.get(person_id, {})
        task_mask = TaskMask(
            box=norm_box is not None,
            pose=_has_pose(pose),
            attr=bool(attr),
            caption=bool(record.get("caption") or record.get("target_text")),
        )
        slots.append(
            PersonSlot(
                slot_id=index,
                box_xyxy=norm_box,
                pose17_xyv=pose if _has_pose(pose) else None,
                attributes=attr,
                caption=record.get("caption"),
                task_mask=task_mask,
            )
        )
    return slots


def build_human_sample(record: Dict[str, Any], image_root: Optional[str | Path] = None) -> HumanSample:
    root = Path(image_root) if image_root else None
    raw = str(record.get("image_abs") or record.get("image_path") or record.get("image", ""))
    path = Path(raw)
    if not path.is_absolute() and root is not None:
        path = root / path
    return HumanSample(
        sample_id=str(record.get("id", "")),
        image_path=str(path),
        width=int(record.get("width", 0) or 0),
        height=int(record.get("height", 0) or 0),
        slots=_build_slots(record),
        raw=record,
    )


class HumanJsonlDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        image_root: Optional[str | Path] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        self.records = _read_jsonl(jsonl_path)
        if max_samples is not None:
            self.records = self.records[: int(max_samples)]
        self.image_root = Path(image_root) if image_root else None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image(self, record: Dict[str, Any]) -> str:
        raw = str(record.get("image_abs") or record.get("image_path") or record.get("image", ""))
        path = Path(raw)
        if path.is_absolute() or self.image_root is None:
            return str(path)
        return str(self.image_root / path)

    def __getitem__(self, index: int) -> HumanSample:
        record = self.records[index]
        sample = build_human_sample(record, image_root=self.image_root)
        if not sample.sample_id:
            sample.sample_id = str(index)
        sample.image_path = self._resolve_image(record)
        return sample


def human_collate(samples: Iterable[HumanSample]) -> HumanBatch:
    return HumanBatch(samples=list(samples))
