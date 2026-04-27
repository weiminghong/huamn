import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
SAPIENS_ROOT = ROOT / "sapiens-main"
for extra in (
    SAPIENS_ROOT / "seg",
    SAPIENS_ROOT / "engine",
    SAPIENS_ROOT / "cv",
    SAPIENS_ROOT / "pretrain",
):
    extra_str = str(extra)
    if extra_str not in sys.path:
        sys.path.insert(0, extra_str)

import torchvision  # noqa: E402
from mmengine.model import revert_sync_batchnorm  # noqa: E402
from mmseg.apis import inference_model, init_model  # noqa: E402

torchvision.disable_beta_transforms_warning()


SAPIENS_CLASSES: List[str] = [
    "Background",
    "Apparel",
    "Face_Neck",
    "Hair",
    "Left_Foot",
    "Left_Hand",
    "Left_Lower_Arm",
    "Left_Lower_Leg",
    "Left_Shoe",
    "Left_Sock",
    "Left_Upper_Arm",
    "Left_Upper_Leg",
    "Lower_Clothing",
    "Right_Foot",
    "Right_Hand",
    "Right_Lower_Arm",
    "Right_Lower_Leg",
    "Right_Shoe",
    "Right_Sock",
    "Right_Upper_Arm",
    "Right_Upper_Leg",
    "Torso",
    "Upper_Clothing",
    "Lower_Lip",
    "Upper_Lip",
    "Lower_Teeth",
    "Upper_Teeth",
    "Tongue",
]

BAD_IMAGE_LIST_PATH = Path("/home/weiminghong/human/stage0/tools/dir_bad_images_list.txt")
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _append_bad_image_path(image_path: Path, reason: str) -> None:
    BAD_IMAGE_LIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with BAD_IMAGE_LIST_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{str(image_path)}\t{reason}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export Sapiens parsing pseudo labels from person_with_instances.json."
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "sapiens-main"
            / "seg"
            / "configs"
            / "sapiens_seg"
            / "goliath"
            / "sapiens_1b_goliath-1024x768.py"
        ),
        help="Sapiens segmentation config path.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/share2/program/weiminghong/coco/checkpoints/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151.pth",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--input-dir",
        default="/share2/program/weiminghong/coco/coco_person/images",
        help="Directory containing input images.",
    )
    parser.add_argument(
        "--ann-json",
        default="/share2/program/weiminghong/coco/coco_person/person_with_instances.json",
        help="Annotation JSON with images[].instance[].",
    )
    parser.add_argument(
        "--output-root",
        default="/share2/program/weiminghong/coco/coco_person/parsing_sapiens",
        help="Output directory containing parsing_labels and parsing.json.",
    )
    parser.add_argument("--device", default="cuda:0", help="Device, e.g. cuda:0 or cpu")
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=24,
        help="Drop an instance if clipped parsing pixels are below this threshold.",
    )
    parser.add_argument(
        "--mask-dilate-k",
        type=int,
        default=3,
        help="Dilation kernel size for segmentation instance masks.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.05,
        help="Drop instance when clipped parsing area / instance mask area is below this value.",
    )
    parser.add_argument(
        "--allow-bbox-fallback",
        action="store_true",
        help="Fallback to bbox mask when segmentation polygon is invalid/missing.",
    )
    parser.add_argument(
        "--keep-empty-images",
        action="store_true",
        help="Keep images with empty person_ids in final parsing.json.",
    )
    parser.add_argument(
        "--verify-image-load",
        action="store_true",
        help="Verify image can be decoded by OpenCV before model inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Chunk size for processing images. Inference is still per-image for stability.",
    )
    parser.add_argument(
        "--scan-missing-only",
        action="store_true",
        help="Only process images missing labels by scanning input-dir and parsing_labels, without using ann-json instances.",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only scan and report missing images (works with --scan-missing-only).",
    )
    parser.add_argument(
        "--missing-list-path",
        default="",
        help="Optional path to save missing image absolute paths when --scan-missing-only is enabled.",
    )
    parser.add_argument(
        "--process-missing-list",
        action="store_true",
        help="Process images from --missing-list-file and save labels to parsing_labels.",
    )
    parser.add_argument(
        "--missing-list-file",
        default="",
        help="Text file containing image absolute paths (one per line).",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("images", [])


def _scan_missing_images(input_dir: Path, labels_dir: Path) -> List[Path]:
    image_paths = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTS]
    done_stems = {p.stem for p in labels_dir.glob("*.png")}
    missing = [p for p in image_paths if p.stem not in done_stems]
    missing.sort(key=lambda p: p.name)
    return missing


def _load_missing_list_file(path: Path) -> List[Path]:
    if not path.exists():
        raise FileNotFoundError(f"missing list file not found: {path}")
    items: List[Path] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s in seen:
                continue
            seen.add(s)
            items.append(Path(s))
    return items


def polygon_to_mask(segmentation: List, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(segmentation, list) or len(segmentation) == 0:
        return mask

    for poly in segmentation:
        if not isinstance(poly, list) or len(poly) < 6 or len(poly) % 2 != 0:
            continue
        pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        pts_i = np.round(pts).astype(np.int32)
        cv2.fillPoly(mask, [pts_i], color=1)
    return mask


def bbox_to_mask(bbox: List[float], height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    if not isinstance(bbox, list) or len(bbox) != 4:
        return mask
    x, y, w, h = bbox
    x1 = max(0, int(np.floor(x)))
    y1 = max(0, int(np.floor(y)))
    x2 = min(width - 1, int(np.ceil(x + w)))
    y2 = min(height - 1, int(np.ceil(y + h)))
    if x2 <= x1 or y2 <= y1:
        return mask
    mask[y1 : y2 + 1, x1 : x2 + 1] = 1
    return mask


def build_instance_mask(
    instance: Dict,
    height: int,
    width: int,
    dilate_k: int,
    allow_bbox_fallback: bool,
) -> np.ndarray:
    mask = polygon_to_mask(instance.get("segmentation", []), height, width)
    if mask.sum() == 0 and allow_bbox_fallback:
        mask = bbox_to_mask(instance.get("bbox", []), height, width)

    if mask.sum() == 0:
        return mask

    k = max(1, int(dilate_k))
    if k > 1:
        kernel = np.ones((k, k), dtype=np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def clip_parsing_by_mask(
    pred_sem_seg: np.ndarray,
    instance_mask: np.ndarray,
) -> np.ndarray:
    clipped = pred_sem_seg.copy()
    clipped[instance_mask == 0] = 0
    return clipped


def get_instance_id(instance: Dict) -> Optional[int]:
    if "instance_id" in instance and instance["instance_id"] is not None:
        return int(instance["instance_id"])
    if "id" in instance and instance["id"] is not None:
        return int(instance["id"])
    return None


def _check_image_readable(image_path: Path) -> Optional[str]:
    """Return None when image looks readable, otherwise return a short reason."""
    if not image_path.exists():
        return "missing_file"
    try:
        if image_path.stat().st_size <= 0:
            return "empty_file"
    except Exception:
        return "stat_failed"
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return "imdecode_failed"
    return None


def process_image_record(
    record: Dict,
    image_path: Path,
    pred_sem_seg: np.ndarray,
    args: argparse.Namespace,
) -> Dict:
    height, width = pred_sem_seg.shape
    final_seg = np.zeros_like(pred_sem_seg, dtype=np.uint8)
    person_ids: List[int] = []

    instances = record.get("instance", [])
    for ins in instances:
        instance_id = get_instance_id(ins)
        if instance_id is None:
            continue

        # Hard condition from your requirement
        if int(ins.get("num_keypoints", 0)) == 0:
            continue

        inst_mask = build_instance_mask(
            instance=ins,
            height=height,
            width=width,
            dilate_k=args.mask_dilate_k,
            allow_bbox_fallback=args.allow_bbox_fallback,
        )
        mask_area = int(inst_mask.sum())
        if mask_area == 0:
            continue

        clipped = clip_parsing_by_mask(pred_sem_seg, inst_mask)
        parsed_area = int((clipped > 0).sum())
        if parsed_area < int(args.min_pixels):
            continue

        coverage_ratio = float(parsed_area) / float(mask_area)
        if coverage_ratio < float(args.min_coverage):
            continue

        final_seg[clipped > 0] = clipped[clipped > 0]
        person_ids.append(instance_id)

    unique_person_ids = sorted(set(person_ids))
    return {
        "image_id": record.get("id"),
        "file_name": record.get("file_name"),
        "img_path": str(image_path),
        "width": int(record.get("width", width)),
        "height": int(record.get("height", height)),
        "person_ids": unique_person_ids,
        "final_seg": final_seg,
    }


def run_inference(args: argparse.Namespace) -> Dict:
    input_dir = Path(args.input_dir)
    ann_json_path = Path(args.ann_json)
    output_root = Path(args.output_root)

    labels_dir = output_root / "parsing_labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    skipped_log_path = output_root / "skipped_images.jsonl"

    if args.process_missing_list:
        missing_list_file = Path(str(args.missing_list_file).strip())
        if not str(args.missing_list_file).strip():
            raise ValueError("--missing-list-file is required when --process-missing-list is enabled.")

        target_images = _load_missing_list_file(missing_list_file)
        model = init_model(args.config, args.checkpoint, device=args.device)
        if args.device == "cpu":
            model = revert_sync_batchnorm(model)

        outputs: List[Dict] = []
        skipped_count = 0
        batch_size = max(1, int(args.batch_size))

        for start in tqdm(range(0, len(target_images), batch_size), desc="Sapiens parsing export (missing-list)"):
            batch_paths = target_images[start : start + batch_size]
            for image_path in batch_paths:
                file_name = image_path.name

                if args.verify_image_load:
                    bad_reason = _check_image_readable(image_path)
                    if bad_reason is not None:
                        skipped_count += 1
                        _append_bad_image_path(image_path, bad_reason)
                        with skipped_log_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "file_name": file_name,
                                        "image_id": None,
                                        "image_path": str(image_path),
                                        "reason": bad_reason,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        continue
                elif not image_path.exists():
                    skipped_count += 1
                    _append_bad_image_path(image_path, "missing_file")
                    with skipped_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "file_name": file_name,
                                    "image_id": None,
                                    "image_path": str(image_path),
                                    "reason": "missing_file",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                try:
                    result = inference_model(model, str(image_path))
                    pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                except Exception as e:
                    skipped_count += 1
                    tb = traceback.format_exc().splitlines()
                    err_msg = str(e)
                    lower_err = err_msg.lower()
                    if (
                        "imdecode" in lower_err
                        or "buf.empty" in lower_err
                        or "cannot decode" in lower_err
                        or "decode" in lower_err
                    ):
                        _append_bad_image_path(image_path, "inference_decode_failed")
                    with skipped_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "file_name": file_name,
                                    "image_id": None,
                                    "image_path": str(image_path),
                                    "reason": "inference_exception",
                                    "error": err_msg,
                                    "traceback_last": tb[-1] if tb else "",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                seg_name = f"{image_path.stem}.png"
                seg_path = labels_dir / seg_name
                cv2.imwrite(str(seg_path), pred_sem_seg)
                h, w = pred_sem_seg.shape[:2]
                outputs.append(
                    {
                        "id": None,
                        "file_name": file_name,
                        "img_path": str(image_path),
                        "seg_map_path": str(seg_path),
                        "width": int(w),
                        "height": int(h),
                        "person_ids": [],
                    }
                )

        payload = {
            "total_count": len(outputs),
            "classes": SAPIENS_CLASSES,
            "meta": {
                "format": "sapiens_semantic_missing_list_only",
                "model_name": "sapiens_1b_goliath",
                "config": args.config,
                "checkpoint": args.checkpoint,
                "input_dir": str(input_dir),
                "ann_json": "",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "process_missing_list": True,
                "missing_list_file": str(missing_list_file),
                "listed_count": len(target_images),
                "verify_image_load": bool(args.verify_image_load),
                "batch_size": int(batch_size),
                "skipped_count": int(skipped_count),
                "skipped_log_path": str(skipped_log_path),
            },
            "images": outputs,
        }

        json_path = output_root / "parsing.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload

    if args.scan_missing_only:
        batch_size = max(1, int(args.batch_size))
        missing_images = _scan_missing_images(input_dir=input_dir, labels_dir=labels_dir)

        missing_list_path = (
            Path(args.missing_list_path)
            if str(args.missing_list_path).strip()
            else (output_root / "missing_images.txt")
        )
        missing_list_path.parent.mkdir(parents=True, exist_ok=True)
        with missing_list_path.open("w", encoding="utf-8") as f:
            for p in missing_images:
                f.write(str(p) + "\n")

        if args.stats_only:
            payload = {
                "total_count": 0,
                "classes": SAPIENS_CLASSES,
                "meta": {
                    "format": "sapiens_semantic_scan_missing_only",
                    "model_name": "sapiens_1b_goliath",
                    "config": args.config,
                    "checkpoint": args.checkpoint,
                    "input_dir": str(input_dir),
                    "ann_json": "",
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "scan_missing_only": True,
                    "stats_only": True,
                    "missing_candidates": len(missing_images),
                    "missing_list_path": str(missing_list_path),
                    "verify_image_load": bool(args.verify_image_load),
                    "batch_size": int(batch_size),
                    "skipped_count": 0,
                    "skipped_log_path": str(skipped_log_path),
                },
                "images": [],
            }
            json_path = output_root / "parsing.json"
            with json_path.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            return payload

        model = init_model(args.config, args.checkpoint, device=args.device)
        if args.device == "cpu":
            model = revert_sync_batchnorm(model)

        outputs: List[Dict] = []
        skipped_count = 0

        for start in tqdm(range(0, len(missing_images), batch_size), desc="Sapiens parsing export (scan-missing-only)"):
            batch_paths = missing_images[start : start + batch_size]
            for image_path in batch_paths:
                file_name = image_path.name

                if args.verify_image_load:
                    bad_reason = _check_image_readable(image_path)
                    if bad_reason is not None:
                        skipped_count += 1
                        _append_bad_image_path(image_path, bad_reason)
                        with skipped_log_path.open("a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "file_name": file_name,
                                        "image_id": None,
                                        "image_path": str(image_path),
                                        "reason": bad_reason,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                        continue
                elif not image_path.exists():
                    skipped_count += 1
                    _append_bad_image_path(image_path, "missing_file")
                    with skipped_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "file_name": file_name,
                                    "image_id": None,
                                    "image_path": str(image_path),
                                    "reason": "missing_file",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                try:
                    result = inference_model(model, str(image_path))
                    pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
                except Exception as e:
                    skipped_count += 1
                    tb = traceback.format_exc().splitlines()
                    err_msg = str(e)
                    lower_err = err_msg.lower()
                    if (
                        "imdecode" in lower_err
                        or "buf.empty" in lower_err
                        or "cannot decode" in lower_err
                        or "decode" in lower_err
                    ):
                        _append_bad_image_path(image_path, "inference_decode_failed")
                    with skipped_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "file_name": file_name,
                                    "image_id": None,
                                    "image_path": str(image_path),
                                    "reason": "inference_exception",
                                    "error": err_msg,
                                    "traceback_last": tb[-1] if tb else "",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue

                seg_name = f"{image_path.stem}.png"
                seg_path = labels_dir / seg_name
                cv2.imwrite(str(seg_path), pred_sem_seg)
                h, w = pred_sem_seg.shape[:2]
                outputs.append(
                    {
                        "id": None,
                        "file_name": file_name,
                        "img_path": str(image_path),
                        "seg_map_path": str(seg_path),
                        "width": int(w),
                        "height": int(h),
                        "person_ids": [],
                    }
                )

        payload = {
            "total_count": len(outputs),
            "classes": SAPIENS_CLASSES,
            "meta": {
                "format": "sapiens_semantic_scan_missing_only",
                "model_name": "sapiens_1b_goliath",
                "config": args.config,
                "checkpoint": args.checkpoint,
                "input_dir": str(input_dir),
                "ann_json": "",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "scan_missing_only": True,
                "stats_only": False,
                "missing_candidates": len(missing_images),
                "missing_list_path": str(missing_list_path),
                "verify_image_load": bool(args.verify_image_load),
                "batch_size": int(batch_size),
                "skipped_count": int(skipped_count),
                "skipped_log_path": str(skipped_log_path),
            },
            "images": outputs,
        }

        json_path = output_root / "parsing.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
    images = load_annotations(ann_json_path)
    if not images:
        raise ValueError(f"No images found in annotation json: {ann_json_path}")

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)
    outputs: List[Dict] = []
    skipped_count = 0
    batch_size = max(1, int(args.batch_size))
    for start in tqdm(range(0, len(images), batch_size), desc="Sapiens parsing export"):
        batch_recs = images[start : start + batch_size]
        for rec in batch_recs:
            file_name = rec.get("file_name")
            if not file_name:
                skipped_count += 1
                with skipped_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "file_name": None,
                                "image_id": rec.get("id"),
                                "reason": "missing_file_name",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            image_path = input_dir / file_name

            if args.verify_image_load:
                bad_reason = _check_image_readable(image_path)
                if bad_reason is not None:
                    skipped_count += 1
                    _append_bad_image_path(image_path, bad_reason)
                    with skipped_log_path.open("a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                {
                                    "file_name": file_name,
                                    "image_id": rec.get("id"),
                                    "image_path": str(image_path),
                                    "reason": bad_reason,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                    continue
            elif not image_path.exists():
                skipped_count += 1
                _append_bad_image_path(image_path, "missing_file")
                with skipped_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "file_name": file_name,
                                "image_id": rec.get("id"),
                                "image_path": str(image_path),
                                "reason": "missing_file",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            try:
                result = inference_model(model, str(image_path))
                pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)
            except Exception as e:
                skipped_count += 1
                tb = traceback.format_exc().splitlines()
                err_msg = str(e)
                lower_err = err_msg.lower()
                if (
                    "imdecode" in lower_err
                    or "buf.empty" in lower_err
                    or "cannot decode" in lower_err
                    or "decode" in lower_err
                ):
                    _append_bad_image_path(image_path, "inference_decode_failed")
                with skipped_log_path.open("a", encoding="utf-8") as f:
                    f.write(
                        json.dumps(
                            {
                                "file_name": file_name,
                                "image_id": rec.get("id"),
                                "image_path": str(image_path),
                                "reason": "inference_exception",
                                "error": err_msg,
                                "traceback_last": tb[-1] if tb else "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue

            item = process_image_record(
                record=rec,
                image_path=image_path,
                pred_sem_seg=pred_sem_seg,
                args=args,
            )

            if len(item["person_ids"]) == 0 and not args.keep_empty_images:
                continue

            seg_name = f"{Path(file_name).stem}.png"
            seg_path = labels_dir / seg_name

            # Save single-channel class-id mask (uint8), used directly as GT label.
            cv2.imwrite(str(seg_path), item["final_seg"])

            outputs.append(
                {
                    "id": item["image_id"],
                    "file_name": item["file_name"],
                    "img_path": item["img_path"],
                    "seg_map_path": str(seg_path),
                    "width": item["width"],
                    "height": item["height"],
                    "person_ids": item["person_ids"],
                }
            )

    payload = {
        "total_count": len(outputs),
        "classes": SAPIENS_CLASSES,
        "meta": {
            "format": "sapiens_semantic_with_person_ids",
            "model_name": "sapiens_1b_goliath",
            "config": args.config,
            "checkpoint": args.checkpoint,
            "input_dir": str(input_dir),
            "ann_json": str(ann_json_path),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "min_pixels": int(args.min_pixels),
            "mask_dilate_k": int(args.mask_dilate_k),
            "min_coverage": float(args.min_coverage),
            "allow_bbox_fallback": bool(args.allow_bbox_fallback),
            "keep_empty_images": bool(args.keep_empty_images),
            "verify_image_load": bool(args.verify_image_load),
            "batch_size": int(batch_size),
            "skipped_count": int(skipped_count),
            "skipped_log_path": str(skipped_log_path),
        },
        "images": outputs,
    }

    json_path = output_root / "parsing.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload

def main() -> None:
    args = parse_args()
    payload = run_inference(args)
    print(f"Saved parsing labels and parsing.json to: {args.output_root}")
    print(f"Total kept images: {payload['total_count']}")
    print(f"Skipped images: {payload['meta'].get('skipped_count', 0)}")


if __name__ == "__main__":
    main()

















