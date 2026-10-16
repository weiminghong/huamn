import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1.eval_stage1_v1 import load_resume_results


def test_load_resume_results_recovers_nul_prefixed_json_and_deduplicates(tmp_path):
    inference = tmp_path / "inference.jsonl"
    first = {"image_id": 11, "schema_ok": True, "records": []}
    recovered = {"image_id": 12, "schema_ok": True, "records": [{"bbox": [1, 2, 3, 4]}]}
    duplicate = {"image_id": 11, "schema_ok": False, "records": [{"bbox": [9, 9, 9, 9]}]}
    inference.write_bytes(
        (json.dumps(first) + "\n").encode("utf-8")
        + b"\x00\x00\x00"
        + (json.dumps(recovered) + "\n").encode("utf-8")
        + b"not-json\n"
        + (json.dumps(duplicate) + "\n").encode("utf-8")
    )

    results, audit = load_resume_results(inference)

    assert list(results) == [11, 12]
    assert results[11] == first
    assert results[12] == recovered
    assert audit == {
        "path": str(inference),
        "lines": 4,
        "valid": 2,
        "recovered": 1,
        "invalid": 1,
        "duplicates": 1,
    }


def test_load_resume_results_missing_file_is_empty(tmp_path):
    inference = tmp_path / "missing.jsonl"
    results, audit = load_resume_results(inference)
