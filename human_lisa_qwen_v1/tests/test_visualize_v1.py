import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.official_eval_v1 import encode_binary_mask
from human_lisa_qwen_v1.visualize_v1 import (
    render_prediction_overlay,
    render_probe_overlays,
)


def _person_record():
    keypoints = []
    for joint in range(17):
        keypoints.extend([20.0 + joint, 30.0 + joint, 2.0])
    mask = np.zeros((80, 100), dtype=np.uint8)
    mask[20:70, 10:60] = 1
    return {
        "image_id": 1,
        "category_id": 1,
        "person_id": 1,
        "bbox": [10.0, 20.0, 50.0, 50.0],
        "score": 0.75,
        "gender": 0,
        "age": 30,
        "keypoints": keypoints,
        "segmentation": encode_binary_mask(mask),
    }


def test_render_prediction_overlay_draws_all_tasks(tmp_path):
    image = Image.new("RGB", (100, 80), color=(240, 240, 240))
    output = tmp_path / "overlay.png"
    metadata = render_prediction_overlay(
        image,
        [_person_record()],
        output,
        title="step=200 image=1",
    )
    assert output.is_file()
    rendered = Image.open(output)
    assert rendered.size == image.size
    assert np.asarray(rendered).var() > 0
    assert metadata == {"persons": 1, "boxes": 1, "poses": 1, "masks": 1}


def test_render_prediction_overlay_handles_empty_predictions(tmp_path):
    image = Image.new("RGB", (64, 48), color="white")
    output = tmp_path / "empty.png"
    metadata = render_prediction_overlay(image, [], output)
    assert output.is_file()
    assert Image.open(output).size == image.size
    assert metadata == {"persons": 0, "boxes": 0, "poses": 0, "masks": 0}


def test_render_probe_overlays_writes_one_image_per_result(tmp_path):
    samples = [
        {"id": 11, "image": Image.new("RGB", (100, 80), color="white")}
    ]
    results = [
        {
            "image_id": 11,
            "schema_ok": True,
            "health": {"ok": True},
            "records": [_person_record()],
        }
    ]
    summary = render_probe_overlays(samples, results, tmp_path, step=200)
    assert summary["images"] == 1
    assert summary["persons"] == 1
    assert summary["files"] == ["step-200-image-11.png"]
    assert (tmp_path / "step-200-image-11.png").is_file()
