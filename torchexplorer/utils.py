from __future__ import annotations

from typing import Any, Iterator

def enumerate_not_none(iterable: Iterator[Any]) -> Iterator[tuple[int, Any]]:
    for i, x in enumerate(iterable):
        if x is not None:
            yield i, x