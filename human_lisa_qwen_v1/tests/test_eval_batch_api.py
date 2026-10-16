import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from human_lisa_qwen_v1 import eval_lisa_qwen_v1 as ev


def test_parse_args_accepts_batch_size():
    with patch.object(sys, "argv", ["eval_lisa_qwen_v1.py", "--batch-size", "4"]):
        args = ev.parse_args()
    assert args.batch_size == 4


def test_iter_batches_keeps_tail_batch():
    rows = [{"id": i} for i in range(5)]
    batches = list(ev.iter_batches(rows, 2))
    assert [[r["id"] for r in batch] for batch in batches] == [[0, 1], [2, 3], [4]]


if __name__ == "__main__":
    test_parse_args_accepts_batch_size()
    test_iter_batches_keeps_tail_batch()


def test_parse_prediction_json_rejects_out_of_schema_values():
    bad = '{"persons":[{"person_id":-1,"box":"<box>","pose":"<pose>","seg":null,"gender":-1,"age":-1,"score":1.2}]}'
    try:
        ev.parse_prediction_json(bad)
    except ValueError:
        return
    raise AssertionError("parse_prediction_json accepted out-of-schema values")
