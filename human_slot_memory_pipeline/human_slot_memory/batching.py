from __future__ import annotations

from typing import Iterable, List, Sequence, TypeVar

T = TypeVar("T")


def batched(items: Sequence[T] | Iterable[T], batch_size: int) -> List[List[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not isinstance(items, list):
        items = list(items)
    return [items[index:index + batch_size] for index in range(0, len(items), batch_size)]
