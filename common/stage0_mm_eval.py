import os.path as osp
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from .io_utils import load_jsonl
from box_task.metrics import evaluate_box_records


def add_sapiens_python_paths(repo_root: Optional[str] = None) -> None:
    """Add local sapiens source trees to sys.path for mmdet/mmpose imports."""
    if repo_root is None:
        repo_root = osp.abspath(osp.join(osp.dirname(__file__), '..'))

    sapiens_root = osp.join(repo_root, 'sapiens-main')
    extra_paths = [
        osp.join(sapiens_root, 'det'),
        osp.join(sapiens_root, 'pose'),
        osp.join(sapiens_root, 'seg'),
        osp.join(sapiens_root, 'engine'),
        osp.join(sapiens_root, 'cv'),
        osp.join(sapiens_root, 'pretrain'),
    ]
    for p in extra_paths:
        if osp.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)


@dataclass
class Stage0Sample:
    sid: str
    image_path: str
    image: Image.Image
    gt_box: List[Any]
    gt_pose: List[Any]


class Stage0BoxPoseDataset(Dataset):
    """Stage0 unified jsonl loader for box+pose evaluation.

    Record schema:
      - id: str/int (optional)
      - image: str (required, absolute or relative)
      - box: list (optional)
      - pose: list (optional)
    """

    def __init__(
        self,
        jsonl_path: str,
        image_root: Optional[str] = None,
        max_samples: Optional[int] = None,
    ) -> None:
        if not jsonl_path:
            raise ValueError('--jsonl is empty. Please provide a valid jsonl path.')
        self.records = load_jsonl(jsonl_path)
        if max_samples is not None:
            self.records = self.records[:max_samples]
        self.image_root = image_root if image_root else None

    def __len__(self) -> int:
        return len(self.records)

    def _resolve_image_path(self, image_path: str) -> str:
        p = Path(image_path)
        if p.is_absolute():
            return str(p)
        if self.image_root is not None:
            return str(Path(self.image_root) / image_path)
        return str(p)

    def __getitem__(self, idx: int) -> Stage0Sample:
        rec = self.records[idx]
        sid = str(rec.get('id', idx))
        image_rel = rec.get('image', '')
        if not isinstance(image_rel, str) or not image_rel:
            raise ValueError(f'Record {idx} has invalid image field: {image_rel}')

        image_path = self._resolve_image_path(image_rel)
        if not osp.isfile(image_path):
            raise FileNotFoundError(f'Image not found: {image_path}')

        image = Image.open(image_path).convert('RGB')
        gt_box = rec.get('box', []) if isinstance(rec.get('box', []), list) else []
        gt_pose = rec.get('pose', []) if isinstance(rec.get('pose', []), list) else []
        return Stage0Sample(sid=sid, image_path=image_path, image=image, gt_box=gt_box, gt_pose=gt_pose)


def collate_identity(batch: List[Stage0Sample]) -> List[Stage0Sample]:
    return batch


def _to_xyxy(box_obj: Any) -> Optional[List[float]]:
    if isinstance(box_obj, dict):
        box = box_obj.get('bbox_xyxy', box_obj.get('bbox', None))
    else:
        box = box_obj
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(box[0]), float(box[1]), float(box[2]), float(box[3])]
    except Exception:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def evaluate_box_with_mmdet_map(
    records: List[Dict[str, Any]],
    iou_thr: float = 0.5,
    nproc: int = 1,
) -> Dict[str, float]:
    """Evaluate box mAP via mmdet.evaluation.functional.eval_map.

    records item format:
      {
        "gt_boxes": list of gt boxes,
        "pred_boxes": list of predicted boxes (dict with bbox_xyxy, score or plain box)
      }
    """
    add_sapiens_python_paths()

    try:
        from mmdet.evaluation.functional.mean_ap import eval_map  # type: ignore
    except Exception:
        fallback_records = []
        for i, rec in enumerate(records):
            fallback_records.append(
                {
                    'id': i,
                    'valid_json': True,
                    'gt_struct': {'box': rec.get('gt_boxes', [])},
                    'pred_struct': {'box': rec.get('pred_boxes', [])},
                }
            )
        simple = evaluate_box_records(fallback_records, iou_thr=iou_thr)
        return {
            'box_mAP_iou50_mmdet': 0.0,
            'box_precision_iou50_fallback': simple.get('box_precision_iou50', 0.0),
            'box_recall_iou50_fallback': simple.get('box_recall_iou50', 0.0),
            'box_f1_iou50_fallback': simple.get('box_f1_iou50', 0.0),
            'box_eval_backend': 0.0,
        }

    det_results: List[List[np.ndarray]] = []
    annotations: List[Dict[str, np.ndarray]] = []

    for rec in records:
        gt_boxes = [_to_xyxy(x) for x in rec.get('gt_boxes', [])]
        gt_boxes = [b for b in gt_boxes if b is not None]

        gt_np = np.asarray(gt_boxes, dtype=np.float32) if gt_boxes else np.zeros((0, 4), dtype=np.float32)
        labels_np = np.zeros((gt_np.shape[0],), dtype=np.int64)
        ann = {
            'bboxes': gt_np,
            'labels': labels_np,
            'bboxes_ignore': np.zeros((0, 4), dtype=np.float32),
            'labels_ignore': np.zeros((0,), dtype=np.int64),
        }
        annotations.append(ann)

        pred_rows = []
        pred_boxes = rec.get('pred_boxes', [])
        if isinstance(pred_boxes, list):
            for p in pred_boxes:
                box = _to_xyxy(p)
                if box is None:
                    continue
                score = 1.0
                if isinstance(p, dict):
                    try:
                        score = float(p.get('score', 1.0))
                    except Exception:
                        score = 1.0
                pred_rows.append(box + [score])

        pred_np = (
            np.asarray(pred_rows, dtype=np.float32)
            if pred_rows
            else np.zeros((0, 5), dtype=np.float32)
        )
        det_results.append([pred_np])

    if len(det_results) == 0:
        return {
            'box_mAP_iou50_mmdet': 0.0,
            'box_eval_backend': 1.0,
        }

    mean_ap, _ = eval_map(
        det_results=det_results,
        annotations=annotations,
        iou_thr=float(iou_thr),
        nproc=max(1, int(nproc)),
    )
    return {
        'box_mAP_iou50_mmdet': float(mean_ap),
        'box_eval_backend': 1.0,
    }


def _normalize_pose_item(pose_obj: Any, image_size: Tuple[int, int]) -> Optional[List[List[float]]]:
    width, height = image_size
    if isinstance(pose_obj, dict):
        kpts = pose_obj.get('keypoints', [])
    elif isinstance(pose_obj, list):
        kpts = pose_obj
    else:
        return None

    if not isinstance(kpts, list):
        return None

    out = []
    for kp in kpts:
        if not isinstance(kp, (list, tuple)) or len(kp) < 2:
            continue
        try:
            x = float(kp[0])
            y = float(kp[1])
            v = float(kp[2]) if len(kp) > 2 else 2.0
        except Exception:
            continue
        x = max(0.0, min(x, width - 1))
        y = max(0.0, min(y, height - 1))
        out.append([x, y, v])
    if len(out) == 0:
        return None
    return out


def evaluate_pose_with_mmpose(
    records: List[Dict[str, Any]],
    pck_thr: float = 0.2,
) -> Dict[str, float]:
    """Evaluate pose using mmpose functional metrics (PCK + EPE).

    records item format:
      {
        "image_size": [W, H],
        "gt_poses": list,
        "pred_poses": list,
        "gt_boxes": list (optional, for normalization)
      }
    """
    add_sapiens_python_paths()

    try:
        from mmpose.evaluation.functional.keypoint_eval import (  # type: ignore
            keypoint_epe,
            keypoint_pck_accuracy,
        )
    except Exception:
        return {
            'pose_pck_mmpose': 0.0,
            'pose_epe_mmpose': 0.0,
            'pose_eval_backend': 0.0,
        }

    pck_num = 0.0
    pck_den = 0.0
    epe_num = 0.0
    epe_den = 0.0

    for rec in records:
        image_size = rec.get('image_size', [1, 1])
        if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
            image_size = [1, 1]
        width, height = int(image_size[0]), int(image_size[1])

        gt_poses = rec.get('gt_poses', [])
        pred_poses = rec.get('pred_poses', [])
        gt_boxes = rec.get('gt_boxes', [])

        if not isinstance(gt_poses, list) or not isinstance(pred_poses, list):
            continue

        pair_n = min(len(gt_poses), len(pred_poses))
        for i in range(pair_n):
            gt_kpts = _normalize_pose_item(gt_poses[i], (width, height))
            pred_kpts = _normalize_pose_item(pred_poses[i], (width, height))
            if gt_kpts is None or pred_kpts is None:
                continue

            k = min(len(gt_kpts), len(pred_kpts))
            if k <= 0:
                continue

            gt_arr = np.asarray([gt_kpts[:k]], dtype=np.float32)
            pred_arr = np.asarray([pred_kpts[:k]], dtype=np.float32)
            mask = np.asarray([[kp[2] > 0 for kp in gt_kpts[:k]]], dtype=bool)

            if mask.sum() <= 0:
                continue

            norm_w = float(width)
            norm_h = float(height)
            box = _to_xyxy(gt_boxes[i]) if isinstance(gt_boxes, list) and i < len(gt_boxes) else None
            if box is not None:
                norm_w = max(1.0, box[2] - box[0])
                norm_h = max(1.0, box[3] - box[1])

            norm_factor = np.asarray([[norm_w, norm_h]], dtype=np.float32)
            _, avg_acc, cnt = keypoint_pck_accuracy(
                pred=pred_arr[:, :, :2],
                gt=gt_arr[:, :, :2],
                mask=mask,
                thr=float(pck_thr),
                norm_factor=norm_factor,
            )
            epe = keypoint_epe(
                pred=pred_arr[:, :, :2],
                gt=gt_arr[:, :, :2],
                mask=mask,
            )

            if cnt > 0:
                pck_num += float(avg_acc) * float(cnt)
                pck_den += float(cnt)
                epe_num += float(epe) * float(cnt)
                epe_den += float(cnt)

    return {
        'pose_pck_mmpose': _safe_div(pck_num, pck_den),
        'pose_epe_mmpose': _safe_div(epe_num, epe_den),
        'pose_eval_backend': 1.0,
    }


def _load_split_tokens(split_txt: Optional[str]) -> Optional[set]:
    if split_txt is None or str(split_txt).strip() == "":
        return None
    p = Path(split_txt)
    if not p.is_file():
        raise FileNotFoundError(f"Split txt not found: {split_txt}")
    tokens = set()
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if not t:
                continue
            tokens.add(t)
            tokens.add(Path(t).stem)
    return tokens


def _is_in_split(file_name: str, image_id: Any, tokens: Optional[set]) -> bool:
    if tokens is None:
        return True
    stem = Path(file_name).stem
    if file_name in tokens or stem in tokens:
        return True
    if image_id is not None and str(image_id) in tokens:
        return True
    return False


def _xyxy_to_xywh(box: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _pose_to_coco_keypoints(pose_obj: Any) -> Optional[List[float]]:
    if isinstance(pose_obj, dict):
        kpts = pose_obj.get("keypoints", [])
    else:
        kpts = pose_obj
    if not isinstance(kpts, list):
        return None
    # Enforce COCO-17 format: pad/truncate to 17.
    out: List[float] = []
    for i in range(17):
        if i < len(kpts) and isinstance(kpts[i], (list, tuple)) and len(kpts[i]) >= 2:
            x = float(kpts[i][0])
            y = float(kpts[i][1])
            v = float(kpts[i][2]) if len(kpts[i]) > 2 else 2.0
            if v < 0.5:
                vv = 0.0
            elif v < 1.5:
                vv = 1.0
            else:
                vv = 2.0
            out.extend([x, y, vv])
        else:
            out.extend([0.0, 0.0, 0.0])
    return out


def evaluate_coco_official_from_predictions(
    pred_lines: List[Dict[str, Any]],
    ann_json: str,
    split_txt: Optional[str] = None,
) -> Dict[str, float]:
    """Official COCO evaluation via pycocotools on bbox and keypoints.

    Args:
        pred_lines: predictions generated by eval script.
        ann_json: person_with_instances.json path.
        split_txt: optional val/train txt path for filtering subset.
    """
    import json

    try:
        from pycocotools.coco import COCO  # type: ignore
        from pycocotools.cocoeval import COCOeval  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "pycocotools is required for official COCO evaluation."
        ) from e

    with Path(ann_json).open("r", encoding="utf-8") as f:
        src = json.load(f)
    images_src = src.get("images", [])
    if not isinstance(images_src, list):
        raise ValueError(f"Invalid ann json format: {ann_json}")

    split_tokens = _load_split_tokens(split_txt)

    images = []
    annotations = []
    ann_id = 1
    valid_image_ids = set()

    for rec in images_src:
        if not isinstance(rec, dict):
            continue
        image_id = rec.get("id", None)
        file_name = str(rec.get("file_name", ""))
        if image_id is None or not file_name:
            continue
        if not _is_in_split(file_name=file_name, image_id=image_id, tokens=split_tokens):
            continue

        valid_image_ids.add(int(image_id))
        images.append(
            {
                "id": int(image_id),
                "file_name": file_name,
                "width": int(rec.get("width", 0)),
                "height": int(rec.get("height", 0)),
            }
        )

        for ins in rec.get("instance", []):
            if not isinstance(ins, dict):
                continue
            bbox = ins.get("bbox", None)
            keypoints = ins.get("keypoints", None)
            num_keypoints = int(ins.get("num_keypoints", 0) or 0)
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            x, y, w, h = [float(v) for v in bbox]
            if w <= 0 or h <= 0:
                continue
            if not isinstance(keypoints, list):
                keypoints = [0.0] * 51
            if len(keypoints) < 51:
                keypoints = keypoints + [0.0] * (51 - len(keypoints))
            elif len(keypoints) > 51:
                keypoints = keypoints[:51]

            annotations.append(
                {
                    "id": ann_id,
                    "image_id": int(image_id),
                    "category_id": 1,
                    "bbox": [x, y, w, h],
                    "area": float(w * h),
                    "iscrowd": 0,
                    "num_keypoints": int(num_keypoints),
                    "keypoints": [float(v) for v in keypoints],
                }
            )
            ann_id += 1

    gt_dataset = {
        "info": {"description": "stage0 coco-official eval"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [
            {
                "id": 1,
                "name": "person",
                "supercategory": "person",
                "keypoints": [
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
                ],
                "skeleton": [],
            }
        ],
    }

    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()

    bbox_preds = []
    kpt_preds = []
    for rec in pred_lines:
        sid = rec.get("id", None)
        if sid is None:
            continue
        try:
            image_id = int(sid)
        except Exception:
            continue
        if image_id not in valid_image_ids:
            continue
        pred = rec.get("prediction_struct_post", {}) or {}

        for b in pred.get("box", []) if isinstance(pred.get("box", []), list) else []:
            xyxy = _to_xyxy(b)
            if xyxy is None:
                continue
            score = 1.0
            if isinstance(b, dict):
                try:
                    score = float(b.get("score", 1.0))
                except Exception:
                    score = 1.0
            bbox_preds.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": _xyxy_to_xywh(xyxy),
                    "score": score,
                }
            )

        for p in pred.get("pose", []) if isinstance(pred.get("pose", []), list) else []:
            kpts = _pose_to_coco_keypoints(p)
            if kpts is None:
                continue
            score = 1.0
            if isinstance(p, dict):
                try:
                    score = float(p.get("score", 1.0))
                except Exception:
                    score = 1.0
            kpt_preds.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "keypoints": kpts,
                    "score": score,
                }
            )

    metrics: Dict[str, float] = {}

    if len(bbox_preds) > 0 and len(images) > 0:
        coco_dt_bbox = coco_gt.loadRes(bbox_preds)
        eval_bbox = COCOeval(coco_gt, coco_dt_bbox, "bbox")
        eval_bbox.evaluate()
        eval_bbox.accumulate()
        eval_bbox.summarize()
        stats = eval_bbox.stats
        metrics.update(
            {
                "coco_bbox_AP": float(stats[0]),
                "coco_bbox_AP50": float(stats[1]),
                "coco_bbox_AP75": float(stats[2]),
                "coco_bbox_APs": float(stats[3]),
                "coco_bbox_APm": float(stats[4]),
                "coco_bbox_APl": float(stats[5]),
            }
        )
    else:
        metrics.update(
            {
                "coco_bbox_AP": 0.0,
                "coco_bbox_AP50": 0.0,
                "coco_bbox_AP75": 0.0,
                "coco_bbox_APs": 0.0,
                "coco_bbox_APm": 0.0,
                "coco_bbox_APl": 0.0,
            }
        )

    if len(kpt_preds) > 0 and len(images) > 0:
        coco_dt_kpt = coco_gt.loadRes(kpt_preds)
        eval_kpt = COCOeval(coco_gt, coco_dt_kpt, "keypoints")
        eval_kpt.evaluate()
        eval_kpt.accumulate()
        eval_kpt.summarize()
        stats = eval_kpt.stats
        metrics.update(
            {
                "coco_kpt_AP": float(stats[0]),
                "coco_kpt_AP50": float(stats[1]),
                "coco_kpt_AP75": float(stats[2]),
                "coco_kpt_APm": float(stats[3]),
                "coco_kpt_APl": float(stats[4]),
                "coco_kpt_AR": float(stats[5]),
                "coco_kpt_AR50": float(stats[6]),
                "coco_kpt_AR75": float(stats[7]),
                "coco_kpt_ARm": float(stats[8]),
                "coco_kpt_ARl": float(stats[9]),
            }
        )
    else:
        metrics.update(
            {
                "coco_kpt_AP": 0.0,
                "coco_kpt_AP50": 0.0,
                "coco_kpt_AP75": 0.0,
                "coco_kpt_APm": 0.0,
                "coco_kpt_APl": 0.0,
                "coco_kpt_AR": 0.0,
                "coco_kpt_AR50": 0.0,
                "coco_kpt_AR75": 0.0,
                "coco_kpt_ARm": 0.0,
                "coco_kpt_ARl": 0.0,
            }
        )

    metrics["coco_num_images"] = float(len(images))
    metrics["coco_num_gt_anns"] = float(len(annotations))
    metrics["coco_num_pred_bbox"] = float(len(bbox_preds))
    metrics["coco_num_pred_kpt"] = float(len(kpt_preds))
    metrics["coco_eval_backend"] = 1.0
    return metrics

