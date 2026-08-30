"""LeCun uniform parameter initialization."""

from __future__ import annotations

from ..tensor import Tensor
from ._utils import DType, Shape
from .variance_scaling import variance_scaling


def lecun_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with uniform variance 1 / fan_in."""
    return variance_scaling(shape, 1.0, "fan_in", "uniform", dtype)


__all__ = ["lecun_uniform"]
