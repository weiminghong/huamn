import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count images with exactly one instance in person_with_instances.json."
    )
    parser.add_argument(
        "--json-path",
        default="/share2/program/weiminghong/coco/coco_person/person_with_instances.json",
        help="Path to person_with_instances.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    json_path = Path(args.json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")

    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    images = payload.get("images", [])
    total_images = len(images)
    one_instance_images = 0

    for item in images:
        instances = item.get("instance", [])
        if isinstance(instances, list) and len(instances) == 1:
            one_instance_images += 1

    print(f"json_path: {json_path}")
    print(f"total_images: {total_images}")
    print(f"one_instance_images: {one_instance_images}")


if __name__ == "__main__":
    main()
