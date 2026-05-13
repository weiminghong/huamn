import json
from pathlib import Path

from human_slot_memory_pipeline.human_slot_memory.stage1_data import (
    build_legacy_train_config,
    convert_raw_annotations_to_stage1_records,
)


def test_convert_raw_annotations_to_stage1_records(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    (image_root / "person.jpg").write_bytes(b"fake")
    ann_path = tmp_path / "val.json"
    payload = {
        "images": [
            {"id": 7, "file_name": "person.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 7,
                "bbox": [64, 48, 128, 240],
                "keypoints": [0, 0, 0] * 17,
                "valid": {"bboxes": 1, "keypoints_2d": 0, "gender": 1, "age": 1},
                "gender": 0,
                "age": 23,
            }
        ],
    }
    ann_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    records, stats = convert_raw_annotations_to_stage1_records(
        annotation_path=ann_path,
        image_root=image_root,
    )

    assert stats["num_images_kept"] == 1
    assert len(records) == 1
    assert records[0]["image_abs"] == str(image_root / "person.jpg")
    assert records[0]["box"][0]["bbox_xyxy"] == [64.0, 48.0, 192.0, 288.0]
    assert records[0]["attribute"][0]["gender"] == 0
    assert records[0]["attribute"][0]["age"] == 23
    assert "<box>[100, 100, 300, 600]</box>" in records[0]["target_text"]
    assert "<gender>0</gender>" in records[0]["target_text"]
    assert "<age>23</age>" in records[0]["target_text"]


def test_build_legacy_train_config_uses_generated_paths(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs"
    cfg = build_legacy_train_config(
        model_name_or_path="/models/Qwen3.5-4B",
        train_jsonl=output_dir / "stage1_sft_data" / "train.jsonl",
        val_jsonl=output_dir / "stage1_sft_data" / "val.jsonl",
        image_root="/images",
        output_dir=output_dir,
        runtime_overrides={"report_to": "none", "bf16": True},
    )

    assert cfg["model_name_or_path"] == "/models/Qwen3.5-4B"
    assert cfg["train_jsonl"].replace("\\", "/").endswith("stage1_sft_data/train.jsonl")
    assert cfg["val_jsonl"].replace("\\", "/").endswith("stage1_sft_data/val.jsonl")
    assert cfg["output_dir"].replace("\\", "/").endswith("outputs/legacy_sft")
    assert cfg["bf16"] is True
    assert cfg["report_to"] == "none"
