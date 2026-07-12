"""Matrix-multiplication public API."""

from typing import Any

from .dot import dot


def matmul(a: Any, b: Any) -> Any:
    """Return the general matrix product of two tensors or Variables."""
    return dot(a, b)


__all__ = ["matmul"]
