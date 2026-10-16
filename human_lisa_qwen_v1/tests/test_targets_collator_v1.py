import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from transformers import AutoProcessor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.collator_v1 import SequenceOverflowError, Stage1Collator
from human_lisa_qwen_v1.schema_v1 import PROMPT_V1, SPECIAL_TOKENS_V1
from human_lisa_qwen_v1.targets_v1 import serialize_target


MODEL_PATH = "/share2/program/weiminghong/model/Qwen2.5-VL-3B-Instruct"


def _instance(person_id, x1, *, gender=-1, gender_valid=False, age=-1, age_valid=False):
    keypoints = np.zeros((17, 3), dtype=np.float32)
    keypoints[:, 0] = np.arange(17) + x1
    keypoints[:, 1] = np.arange(17) + 10
    keypoints[:, 2] = 2
    mask = np.zeros((80, 100), dtype=np.bool_)
    mask[10:40, int(x1) : int(x1) + 20] = True
    return {
        "person_id": person_id,
        "source_ann_id": person_id + 100,
        "source_image_id": 7,
        "bbox_xyxy": [float(x1), 10.0, float(x1 + 20), 40.0],
        "area": 600.0,
        "keypoints": keypoints,
        "pose_valid": True,
        "mask": mask,
        "seg_valid": True,
        "gender": gender,
        "gender_valid": gender_valid,
        "age": float(age),
        "age_valid": age_valid,
    }


def _sample(sample_id, instances):
    return {
        "id": sample_id,
        "file_name": "synthetic.jpg",
        "image_path": "synthetic.jpg",
        "image": Image.new("RGB", (100, 80), color="white"),
        "width": 100,
        "height": 80,
        "instances": instances,
        "skipped": {},
        "flipped": False,
    }


@pytest.fixture(scope="module")
def processor_and_ids():
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer = processor.tokenizer
    existing = list(getattr(tokenizer, "additional_special_tokens", []) or [])
    new_tokens = [token for token in SPECIAL_TOKENS_V1 if token not in existing]
    if new_tokens:
        tokenizer.add_special_tokens(
            {"additional_special_tokens": existing + new_tokens}
        )
    token_ids = {
        name: tokenizer.convert_tokens_to_ids(token)
        for name, token in {
            "box": "<box>",
            "pose": "<pose>",
            "seg": "<seg>",
        }.items()
    }
    return processor, token_ids


def test_serialize_target_tracks_binding_and_missing_attribute_spans():
    target = serialize_target(
        [
            _instance(
                1,
                10,
                gender=0,
                gender_valid=True,
                age=21,
                age_valid=True,
            ),
            _instance(2, 60),
        ],
        width=100,
        height=80,
    )

    assert target.text.count("<box>") == 2
    assert target.text.count("<pose>") == 2
    assert target.text.count("<seg>") == 2
    assert '"score"' not in target.text
    assert target.box_person_indices == [0, 1]
    assert target.pose_person_indices == [0, 1]
    assert target.seg_person_indices == [0, 1]
    assert len(target.missing_attribute_char_spans) == 2
    for start, end, field, person_index in target.missing_attribute_char_spans:
        assert target.text[start:end] == "-1"
        assert field in {"gender", "age"}
        assert person_index == 1
    np.testing.assert_allclose(
        target.box_targets[0], [0.1, 0.125, 0.3, 0.5]
    )


def test_collator_masks_prefix_padding_and_only_missing_values(processor_and_ids):
    processor, token_ids = processor_and_ids
    collator = Stage1Collator(
        processor=processor,
        token_ids=token_ids,
        prompt=PROMPT_V1,
        max_length=5000,
    )
    batch = collator(
        [
            _sample(
                7,
                [
                    _instance(
                        1,
                        10,
                        gender=0,
                        gender_valid=True,
                        age=21,
                        age_valid=True,
                    ),
                    _instance(2, 60),
                ],
            ),
            _sample(8, [_instance(1, 20, gender=1, gender_valid=True)]),
        ]
    )

    assert batch["input_ids"].shape[0] == 2
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["sequence_lengths"].max().item() < 5000
    for row, audit in enumerate(batch["label_audit"]):
        target_start = audit["target_start"]
        assert (batch["labels"][row, :target_start] == -100).all()
        assert audit["masked_attribute_token_positions"]
        for position in audit["masked_attribute_token_positions"]:
            assert batch["labels"][row, position].item() == -100
        assert audit["special_token_counts"] == audit["target_counts"]
    padding = batch["attention_mask"] == 0
    assert (batch["labels"][padding] == -100).all()
    assert (batch["language_loss_weights"][padding] == 0).all()
    assert (batch["language_loss_weights"][batch["labels"] == -100] == 0).all()


def test_collator_rejects_instead_of_truncating(processor_and_ids):
    processor, token_ids = processor_and_ids
    collator = Stage1Collator(
        processor=processor,
        token_ids=token_ids,
        prompt=PROMPT_V1,
        max_length=10,
    )
    with pytest.raises(SequenceOverflowError) as exc_info:
        collator([_sample(991, [_instance(1, 10)])])
    message = str(exc_info.value)
    assert "image_id=991" in message
    assert "max_length=10" in message
    assert "truncation is disabled" in message


def test_schema_prompt_has_no_person_cap_or_text_score():
    assert "maximum person" not in PROMPT_V1.lower()
    assert "max persons" not in PROMPT_V1.lower()
    assert '"score"' not in PROMPT_V1
    assert "JSON only" in PROMPT_V1
