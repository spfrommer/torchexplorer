from __future__ import annotations

import platform
from typing import Any, Iterator, Iterable
import torch

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


def get_device() -> torch.device:
    """Get the best available device (MPS for Mac, CUDA if available, otherwise CPU)"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def device_available() -> bool:
    return torch.backends.mps.is_available() or torch.cuda.is_available()

def get_system() -> str:
    """Get the current operating system."""
    system = platform.system().lower()
    if system == "darwin":
        return "darwin"
    elif system == "linux":
        return "linux"
    else:
        raise ValueError(f"Unsupported system: {system}")