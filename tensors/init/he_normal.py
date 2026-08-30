"""He normal parameter initialization."""

from __future__ import annotations

from ..tensor import Tensor
from ._utils import DType, Shape
from .variance_scaling import variance_scaling


def he_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with normal variance 2 / fan_in."""
    return variance_scaling(shape, 2.0, "fan_in", "normal", dtype)


__all__ = ["he_normal"]
