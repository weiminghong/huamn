import json
from pathlib import Path


ANN_JSON = "/share2/program/weiminghong/coco/coco_person/person_with_instances.json"
TARGET_IMAGE = "000000102684.jpg"


def flat_to_triplets(flat_kpts):
    triplets = []
    if not isinstance(flat_kpts, list):
        return triplets
    n = len(flat_kpts) // 3
    for i in range(n):
        try:
            x = float(flat_kpts[i * 3 + 0])
            y = float(flat_kpts[i * 3 + 1])
            v = float(flat_kpts[i * 3 + 2])
        except Exception:
            x, y, v = 0.0, 0.0, 0.0
        triplets.append([x, y, v])
    return triplets


def main():
    ann_path = Path(ANN_JSON)
    if not ann_path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {ANN_JSON}")

    with ann_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    if not isinstance(images, list):
        raise ValueError(f"Invalid annotation format in: {ANN_JSON}")

    target = None
    for rec in images:
        if isinstance(rec, dict) and str(rec.get("file_name", "")).strip() == TARGET_IMAGE:
            target = rec
            break

    if target is None:
        raise ValueError(f"Image not found in annotations: {TARGET_IMAGE}")

    result = {
        "id": target.get("id"),
        "file_name": target.get("file_name"),
        "num_instances": len(target.get("instance", [])) if isinstance(target.get("instance", []), list) else 0,
        "instances": [],
    }

    for ins in target.get("instance", []):
        if not isinstance(ins, dict):
            continue
        result["instances"].append(
            {
                "instance_id": ins.get("instance_id"),
                "bbox_xywh": ins.get("bbox", []),
                "num_keypoints": ins.get("num_keypoints", 0),
                "keypoints_flat": ins.get("keypoints", []),
                "keypoints_triplets": flat_to_triplets(ins.get("keypoints", [])),
            }
        )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
