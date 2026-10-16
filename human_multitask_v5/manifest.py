from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple

from .schema import summarize, validate_record


def iter_jsonl(path: str | Path) -> Iterator[Dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            validate_record(record)
            yield record


def write_jsonl(path: str | Path, records: Iterable[Mapping]) -> Dict:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    count = 0
    with target.open("x", encoding="utf-8", newline="\n") as handle:
        for record in records:
            validate_record(record)
            line = json.dumps(
                record, ensure_ascii=True, separators=(",", ":")
            ) + "\n"
            handle.write(line)
            digest.update(line.encode("utf-8"))
            count += 1
    return {"records": count, "sha256": digest.hexdigest(), "path": str(target)}


def image_identity(record: Mapping) -> str:
    image = record["image"]
    coco_id = image.get("coco_id")
    if coco_id is not None:
        return f"coco:{int(coco_id)}"
    return (
        f"{record['dataset']}:{image.get('source_image_id', record['uid'])}"
    )


def audit_manifests(paths: Sequence[str | Path]) -> Dict:
    seen_uid: Dict[str, str] = {}
    seen_image: Dict[str, Tuple[str, str]] = {}
    duplicate_uids: List[Dict] = []
    split_leakage: List[Dict] = []
    records: List[Dict] = []
    for path in paths:
        for record in iter_jsonl(path):
            uid = str(record["uid"])
            if uid in seen_uid:
                duplicate_uids.append(
                    {"uid": uid, "first": seen_uid[uid], "second": str(path)}
                )
            else:
                seen_uid[uid] = str(path)
            identity = image_identity(record)
            current = (str(record["dataset"]), str(record["split"]))
            if identity in seen_image and seen_image[identity][1] != current[1]:
                split_leakage.append(
                    {
                        "identity": identity,
                        "first": seen_image[identity],
                        "second": current,
                    }
                )
            else:
                seen_image[identity] = current
            records.append(record)
    summary = asdict(summarize(records))
    summary.update(
        {
            "duplicate_uids": duplicate_uids,
            "split_leakage": split_leakage,
            "valid": not duplicate_uids and not split_leakage,
        }
    )
    return summary


def box_iou(left: Sequence[float], right: Sequence[float]) -> float:
    lx1, ly1, lx2, ly2 = map(float, left)
    rx1, ry1, rx2, ry2 = map(float, right)
    iw = max(0.0, min(lx2, rx2) - max(lx1, rx1))
    ih = max(0.0, min(ly2, ry2) - max(ly1, ry1))
    intersection = iw * ih
    la = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    ra = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    return intersection / max(la + ra - intersection, 1e-12)


def audited_box_matches(
    anchors: Sequence[Mapping],
    additions: Sequence[Mapping],
    *,
    threshold: float = 0.7,
    ambiguity_margin: float = 0.05,
) -> Dict:
    """Conservative instance join for cross-dataset annotations.

    A match is accepted only when it is mutual-best, above threshold, and
    separated from the runner-up by the declared ambiguity margin.
    """
    scores = [
        [box_iou(a["bbox_xyxy"], b["bbox_xyxy"]) for b in additions]
        for a in anchors
    ]
    accepted, rejected = [], []
    for ai, row in enumerate(scores):
        if not row:
            continue
        order = sorted(range(len(row)), key=row.__getitem__, reverse=True)
        bi = order[0]
        best = row[bi]
        runner = row[order[1]] if len(order) > 1 else 0.0
        column = [scores[x][bi] for x in range(len(anchors))]
        mutual = ai == max(range(len(column)), key=column.__getitem__)
        item = {
            "anchor_index": ai,
            "addition_index": bi,
            "iou": best,
            "runner_up_iou": runner,
            "mutual_best": mutual,
        }
        if (
            best >= threshold
            and best - runner >= ambiguity_margin
            and mutual
        ):
            accepted.append(item)
        else:
            rejected.append(item)
    return {
        "accepted": accepted,
        "rejected": rejected,
        "threshold": threshold,
        "ambiguity_margin": ambiguity_margin,
    }
