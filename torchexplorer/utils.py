from __future__ import annotations

from typing import Any, Iterator, Iterable

def iter_not_none(iterable: Iterable[Any]) -> Iterator[Any]:
    for x in iterable:
        if x is not None:
            yield x

def enum_not_none(iterable: Iterable[Any]) -> Iterator[tuple[int, Any]]:
    for i, x in enumerate(iterable):
        if x is not None:
            yield i, x

def interleave(l1: list[Any], l2: list[Any]) -> list[Any]:
    assert len(l1) == len(l2)
    return [x for pair in zip(l1, l2) for x in pair]


def list_add(l1: list[float], l2: list[float]) -> list[float]:
    assert len(l1) == len(l2)
    return [x + y for x, y in zip(l1, l2)]