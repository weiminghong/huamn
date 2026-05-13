import pytest

from human_slot_memory_pipeline.human_slot_memory.batching import batched


def test_batched_splits_evenly() -> None:
    assert batched([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]


def test_batched_keeps_tail_batch() -> None:
    assert batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]


def test_batched_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError):
        batched([1, 2], 0)
