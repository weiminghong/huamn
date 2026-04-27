import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test Sapiens segmentation inference with person_with_instances.json."
    )
    parser.add_argument(
        "--json-path",
        default="/share2/program/weiminghong/coco/coco_person/person_with_instances.json",
        help="Path to person_with_instances.json",
    )
    parser.add_argument(
        "--image-root",
        default="/share2/program/weiminghong/coco/coco_person/images",
        help="Directory containing source images.",
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
        help="Sapiens checkpoint path.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Inference device, e.g. cuda:0 or cpu.",
    )
    parser.add_argument(
        "--file-name",
        default="",
        help="If set, infer this image file name directly (e.g. 000000391895.jpg).",
    )
    parser.add_argument(
        "--image-index",
        type=int,
        default=0,
        help="Fallback image index in json when --file-name is not provided.",
    )
    parser.add_argument(
        "--output-dir",
        default="/share2/program/weiminghong/coco/coco_person/parsing_sapiens_test",
        help="Output directory for test artifacts.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_image_record(images: List[Dict], file_name: str, image_index: int) -> Dict:
    if file_name:
        for rec in images:
            if rec.get("file_name") == file_name:
                return rec
        raise ValueError(f"file_name not found in json: {file_name}")
    if image_index < 0 or image_index >= len(images):
        raise IndexError(f"image_index out of range: {image_index}, total={len(images)}")
    return images[image_index]


def segmentation_to_mask(segmentation: List[List[float]], height: int, width: int) -> np.ndarray:
    """
    将COCO格式的segmentation多边形转换为二值mask

    Args:
        segmentation: COCO格式的多边形列表 [[x1,y1,x2,y2,...], ...]
        height: 图像高度
        width: 图像宽度

    Returns:
        二值mask (H, W)，人体区域为1，背景为0
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in segmentation:
        # 将多边形坐标转换为numpy数组
        poly_np = np.array(poly).reshape(-1, 2).astype(np.int32)
        # 填充多边形区域
        cv2.fillPoly(mask, [poly_np], 1)
    return mask


def evaluate_instance_segmentation(
    pred_seg: np.ndarray,
    gt_mask: np.ndarray,
    instance_id: int,
    expected_class_id: int = 1  # 假设人体类别ID为1
) -> Dict:
    """
    评估单个instance的分割质量

    Args:
        pred_seg: Sapiens预测的分割图 (H, W)
        gt_mask: ground truth mask (H, W)
        instance_id: 实例ID
        expected_class_id: 期望的人体类别ID

    Returns:
        包含IoU、准确率等指标的字典
    """
    # 只在GT mask区域内评估
    pred_in_mask = pred_seg[gt_mask == 1]

    # 统计预测为人体类别的像素数
    pred_positive = (pred_in_mask == expected_class_id).sum()
    total_pixels = gt_mask.sum()

    # 计算准确率
    accuracy = pred_positive / total_pixels if total_pixels > 0 else 0.0

    # 计算IoU
    pred_mask = (pred_seg == expected_class_id).astype(np.uint8)
    intersection = (pred_mask & gt_mask).sum()
    union = (pred_mask | gt_mask).sum()
    iou = intersection / union if union > 0 else 0.0

    return {
        "instance_id": instance_id,
        "total_pixels": int(total_pixels),
        "pred_positive_pixels": int(pred_positive),
        "accuracy": float(accuracy),
        "iou": float(iou),
        "pred_unique_classes": np.unique(pred_in_mask).tolist()
    }


def collect_person_ids(instances: List[Dict]) -> Dict[str, List[int]]:
    all_ids: List[int] = []
    pose_valid_ids: List[int] = []
    for ins in instances:
        pid = ins.get("instance_id")  # 修改为 instance_id
        if pid is None:
            continue
        all_ids.append(int(pid))
        if int(ins.get("num_keypoints", 0)) > 0:
            pose_valid_ids.append(int(pid))
    return {"all_person_ids": all_ids, "pose_valid_person_ids": pose_valid_ids}


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    image_root = Path(args.image_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(json_path)
    images = payload.get("images", [])
    if not images:
        raise ValueError(f"No images found in json: {json_path}")

    record = select_image_record(images, args.file_name, args.image_index)
    file_name = record["file_name"]
    image_path = image_root / file_name
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)

    result = inference_model(model, str(image_path))
    pred_sem_seg = result.pred_sem_seg.data[0].cpu().numpy().astype(np.uint8)

    seg_dir = output_dir / "seg_maps"
    seg_dir.mkdir(parents=True, exist_ok=True)
    seg_path = seg_dir / f"{Path(file_name).stem}.png"
    cv2.imwrite(str(seg_path), pred_sem_seg)

    person_info = collect_person_ids(record.get("instance", []))
    unique_ids = sorted(np.unique(pred_sem_seg).tolist())

    # 评估每个instance的分割质量
    instance_results = []
    for ins in record.get("instance", []):
        instance_id = ins.get("instance_id")
        segmentation = ins.get("segmentation", [])

        if not segmentation or instance_id is None:
            continue

        # 转换为mask
        gt_mask = segmentation_to_mask(
            segmentation,
            pred_sem_seg.shape[0],
            pred_sem_seg.shape[1]
        )

        # 评估该instance
        eval_result = evaluate_instance_segmentation(
            pred_sem_seg,
            gt_mask,
            instance_id
        )

        # 添加额外信息
        eval_result["num_keypoints"] = ins.get("num_keypoints", 0)
        eval_result["bbox"] = ins.get("bbox", [])

        instance_results.append(eval_result)

    summary = {
        "image_id": record.get("id"),
        "file_name": file_name,
        "image_path": str(image_path),
        "seg_map_path": str(seg_path),
        "width": int(record.get("width", pred_sem_seg.shape[1])),
        "height": int(record.get("height", pred_sem_seg.shape[0])),
        "num_instances": len(record.get("instance", [])),
        "all_person_ids": person_info["all_person_ids"],
        "pose_valid_person_ids": person_info["pose_valid_person_ids"],
        "pred_unique_class_ids": unique_ids,
        "instance_evaluations": instance_results,
    }

    summary_path = output_dir / "test_result.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"image_path: {image_path}")
    print(f"seg_map_path: {seg_path}")
    print(f"summary_path: {summary_path}")
    print(f"num_instances: {summary['num_instances']}")
    print(f"pose_valid_person_ids: {summary['pose_valid_person_ids']}")
    print(f"pred_unique_class_ids: {summary['pred_unique_class_ids']}")
    print("\nInstance Evaluations:")
    for eval_res in instance_results:
        print(f"  Instance {eval_res['instance_id']}: "
              f"IoU={eval_res['iou']:.4f}, "
              f"Accuracy={eval_res['accuracy']:.4f}, "
              f"Pixels={eval_res['total_pixels']}, "
              f"Keypoints={eval_res['num_keypoints']}")


if __name__ == "__main__":
    main()
