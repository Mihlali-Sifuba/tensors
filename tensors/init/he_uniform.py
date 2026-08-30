"""He uniform parameter initialization."""

from __future__ import annotations

from ..tensor import Tensor
from ._utils import DType, Shape
from .variance_scaling import variance_scaling


def he_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with uniform variance 2 / fan_in."""
    return variance_scaling(shape, 2.0, "fan_in", "uniform", dtype)


__all__ = ["he_uniform"]
