from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from mmdet.structures.mask import BitmapMasks
from pycocotools import mask as mask_utils

MMDET_MEAN = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32).view(3, 1, 1)
MMDET_STD = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32).view(3, 1, 1)
VALID_ORDER = ("labels", "bboxes", "keypoints_2d", "keypoints_3d", "segmentation", "smpl", "gender", "age")


def _as_bool(value: Any) -> bool:
    return bool(value) and value not in ("0", "false", "False")


def _valid_vector(ann: Dict[str, Any]) -> List[int]:
    valid = ann.get("valid") or {}
    # HumanQueryNet's original parser marks label validity together with bbox validity.
    has_box = _as_bool(valid.get("bboxes", ann.get("bbox") is not None))
    return [
        int(has_box),
        int(has_box),
        int(_as_bool(valid.get("keypoints_2d", False))),
        int(_as_bool(valid.get("keypoints_3d", False))),
        int(_as_bool(valid.get("segmentation", valid.get("mask", False)))),
        int(_as_bool(valid.get("smpl", False))),
        int(_as_bool(valid.get("gender", False))),
        int(_as_bool(valid.get("age", False))),
    ]


def _annotation_has_valid_box(ann: Dict[str, Any], width: int, height: int) -> bool:
    valid = ann.get("valid") or {}
    if valid and not _as_bool(valid.get("bboxes", False)):
        return False
    bbox = ann.get("bbox")
    if not bbox or len(bbox) != 4:
        return False
    x, y, w, h = map(float, bbox)
    inter_w = max(0.0, min(x + w, width) - max(x, 0.0))
    inter_h = max(0.0, min(y + h, height) - max(y, 0.0))
    return inter_w > 0 and inter_h > 0 and w >= 1 and h >= 1 and float(ann.get("area", w * h)) > 0


def _annotation_has_valid_keypoints(ann: Dict[str, Any]) -> bool:
    valid = ann.get("valid") or {}
    if not _as_bool(valid.get("keypoints_2d", False)):
        return False
    keypoints = ann.get("keypoints") or []
    return len(keypoints) >= 51 and any(float(keypoints[i + 2]) > 0 for i in range(0, min(len(keypoints), 51), 3))


def _build_image_index(data: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    by_image: Dict[int, List[Dict[str, Any]]] = {}
    for ann in data.get("annotations", []):
        by_image.setdefault(int(ann["image_id"]), []).append(ann)
    return list(data.get("images", [])), by_image


def _find_image_record(
    images: Sequence[Dict[str, Any]],
    anns_by_image: Dict[int, List[Dict[str, Any]]],
    image_root: Path,
    image_id: Optional[int],
    require_keypoints: bool,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Path]:
    candidates: Iterable[Dict[str, Any]]
    if image_id is not None:
        candidates = [img for img in images if int(img.get("id", img.get("img_id"))) == int(image_id)]
    else:
        candidates = images

    for img in candidates:
        img_id = int(img.get("id", img.get("img_id")))
        width, height = int(img["width"]), int(img["height"])
        anns = [a for a in anns_by_image.get(img_id, []) if _annotation_has_valid_box(a, width, height)]
        if require_keypoints:
            anns = [a for a in anns if _annotation_has_valid_keypoints(a)]
        if not anns:
            continue
        file_name = img.get("file_name") or img.get("filename")
        if not file_name:
            continue
        img_path = image_root / file_name
        if img_path.exists():
            return img, anns, img_path
    raise ValueError("No usable COCO-UniHuman sample found for the requested constraints")


def _resize_image_tensor(img_path: Path, image_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    image = Image.open(img_path).convert("RGB")
    ori_w, ori_h = image.size
    resized = image.resize((image_size, image_size), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1).float()
    return (arr - MMDET_MEAN) / MMDET_STD, (ori_h, ori_w)


def _decode_mask(segmentation: Any, height: int, width: int) -> np.ndarray:
    if not segmentation:
        return np.zeros((height, width), dtype=np.uint8)
    try:
        if isinstance(segmentation, list):
            rles = mask_utils.frPyObjects(segmentation, height, width)
            rle = mask_utils.merge(rles)
        elif isinstance(segmentation, dict) and isinstance(segmentation.get("counts"), list):
            rle = mask_utils.frPyObjects(segmentation, height, width)
        else:
            rle = segmentation
        decoded = mask_utils.decode(rle)
        if decoded.ndim == 3:
            decoded = decoded.any(axis=2)
        return decoded.astype(np.uint8)
    except Exception:
        return np.zeros((height, width), dtype=np.uint8)


def _resize_mask(mask: np.ndarray, image_size: int) -> np.ndarray:
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    image = image.resize((image_size, image_size), Image.NEAREST)
    return (np.asarray(image) > 0).astype(np.uint8)


def _gender_value(raw: Any) -> int:
    if raw is None:
        return -1
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"male", "m", "man", "1"}:
            return 1
        if lowered in {"female", "f", "woman", "0"}:
            return 0
        return -1
    try:
        return int(raw)
    except Exception:
        return -1


def _age_value(raw: Any) -> int:
    if raw is None:
        return -1
    try:
        return int(round(float(raw)))
    except Exception:
        return -1


def _scale_keypoints(ann: Dict[str, Any], sx: float, sy: float, image_size: Optional[int] = None) -> List[float]:
    keypoints = list(ann.get("keypoints") or [])
    if len(keypoints) < 51:
        keypoints = keypoints + [0.0] * (51 - len(keypoints))
    keypoints = keypoints[:51]
    limit = None if image_size is None else max(float(image_size), 1.0)
    out: List[float] = []
    for i in range(0, 51, 3):
        x = float(keypoints[i]) * sx
        y = float(keypoints[i + 1]) * sy
        v = float(keypoints[i + 2])
        if v <= 0:
            x, y = 0.0, 0.0
        elif limit is not None:
            x = max(0.0, min(x, limit))
            y = max(0.0, min(y, limit))
        out.extend([x, y, v])
    return out


def build_humanv2_stage2_batch(
    ann_file: str,
    image_root: str,
    image_size: int = 128,
    max_instances: Optional[int] = None,
    image_id: Optional[int] = None,
    require_keypoints: bool = False,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Build one mmdet3-era HumanQueryNet training batch directly from COCO-UniHuman JSON.

    This intentionally bypasses the old MMDet2 Collect/DataContainer pipeline while preserving
    the target layout expected by HumanQueryNet.forward_train.
    """
    ann_path = Path(ann_file)
    root = Path(image_root)
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    images, anns_by_image = _build_image_index(data)
    img_info, anns, img_path = _find_image_record(images, anns_by_image, root, image_id, require_keypoints)
    if max_instances is not None:
        anns = anns[: int(max_instances)]

    image_size = int(image_size)
    image_tensor, (ori_h, ori_w) = _resize_image_tensor(img_path, image_size)
    sx, sy = image_size / float(ori_w), image_size / float(ori_h)

    bboxes: List[List[float]] = []
    labels: List[int] = []
    attributes: List[List[int]] = []
    keypoints: List[List[float]] = []
    areas: List[float] = []
    valids: List[List[int]] = []
    smpl_pose: List[List[float]] = []
    smpl_betas: List[List[float]] = []
    masks: List[np.ndarray] = []

    for ann in anns:
        x, y, w, h = map(float, ann["bbox"])
        x1 = max(0.0, min(float(image_size), x * sx))
        y1 = max(0.0, min(float(image_size), y * sy))
        x2 = max(0.0, min(float(image_size), (x + w) * sx))
        y2 = max(0.0, min(float(image_size), (y + h) * sy))
        if x2 <= x1 or y2 <= y1:
            continue
        valid = _valid_vector(ann)
        bboxes.append([x1, y1, x2, y2])
        labels.append(0)
        attributes.append([
            _gender_value(ann.get("gender")) if valid[6] else -1,
            _age_value(ann.get("age")) if valid[7] else -1,
        ])
        keypoints.append(_scale_keypoints(ann, sx, sy) if valid[2] else [0.0] * 51)
        areas.append(float(ann.get("area", w * h)) * sx * sy)
        valids.append(valid)
        smpl = ann.get("smpl") if valid[5] else None
        smpl_pose.append(list((smpl or {}).get("pose", [0.0] * 72))[:72])
        smpl_betas.append(list((smpl or {}).get("betas", [0.0] * 10))[:10])
        mask = _decode_mask(ann.get("segmentation") if valid[4] else None, ori_h, ori_w)
        masks.append(_resize_mask(mask, image_size))

    if not bboxes:
        raise ValueError("Selected sample has no valid instances after scaling")

    device = torch.device(device)
    img = image_tensor.unsqueeze(0).to(device)
    gt_bboxes = [torch.tensor(bboxes, dtype=torch.float32, device=device)]
    gt_labels = [torch.tensor(labels, dtype=torch.long, device=device)]
    gt_attributes = [torch.tensor(attributes, dtype=torch.long, device=device)]
    gt_keypoints = [torch.tensor(keypoints, dtype=torch.float32, device=device)]
    gt_areas = [torch.tensor(areas, dtype=torch.float32, device=device)]
    gt_valids = [torch.tensor(valids, dtype=torch.long, device=device)]
    gt_smpl_pose = [torch.tensor(smpl_pose, dtype=torch.float32, device=device)]
    gt_smpl_betas = [torch.tensor(smpl_betas, dtype=torch.float32, device=device)]
    gt_masks = [BitmapMasks(np.stack(masks, axis=0), image_size, image_size)]
    img_meta = {
        "filename": str(img_path),
        "ori_filename": img_path.name,
        "img_shape": (image_size, image_size, 3),
        "ori_shape": (ori_h, ori_w, 3),
        "pad_shape": (image_size, image_size, 3),
        "batch_input_shape": (image_size, image_size),
        "scale_factor": np.array([sx, sy, sx, sy], dtype=np.float32),
        "flip": False,
        "flip_direction": None,
        "img_norm_cfg": {"mean": [123.675, 116.28, 103.53], "std": [58.395, 57.12, 57.375], "to_rgb": True},
    }
    return {
        "img": img,
        "img_metas": [img_meta],
        "gt_bboxes": gt_bboxes,
        "gt_labels": gt_labels,
        "gt_attributes": gt_attributes,
        "gt_keypoints": gt_keypoints,
        "gt_areas": gt_areas,
        "gt_masks": gt_masks,
        "gt_valids": gt_valids,
        "gt_smpl_pose": gt_smpl_pose,
        "gt_smpl_betas": gt_smpl_betas,
        "sample_info": {"img_id": int(img_info.get("id", img_info.get("img_id"))), "img_path": str(img_path), "num_instances": len(bboxes)},
    }
