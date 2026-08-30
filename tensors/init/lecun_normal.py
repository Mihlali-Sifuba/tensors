"""LeCun normal parameter initialization."""

from __future__ import annotations

from ..tensor import Tensor
from ._utils import DType, Shape
from .variance_scaling import variance_scaling


def lecun_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with normal variance 1 / fan_in."""
    return variance_scaling(shape, 1.0, "fan_in", "normal", dtype)


__all__ = ["lecun_normal"]
