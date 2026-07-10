"""Matrix-multiplication public API."""

from typing import Any

from .dot import dot


def matmul(a: Any, b: Any) -> Any:
    """Matrix multiplication for the currently supported 2D inputs."""
    return dot(a, b)


__all__ = ["matmul"]
