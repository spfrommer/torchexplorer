from __future__ import annotations

from typing import Any, Iterator

def iter_not_none(iterable: Iterator[Any]) -> Iterator[Any]:
    for x in iterable:
        if x is not None:
            yield x

def enum_not_none(iterable: Iterator[Any]) -> Iterator[tuple[int, Any]]:
    for i, x in enumerate(iterable):
        if x is not None:
            yield i, x