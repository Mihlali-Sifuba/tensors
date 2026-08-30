"""Xavier uniform parameter initialization."""

from __future__ import annotations

from ..tensor import Tensor
from ._utils import DType, Shape
from .variance_scaling import variance_scaling


def xavier_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with uniform variance 2 / (fan_in + fan_out)."""
    return variance_scaling(shape, 1.0, "fan_avg", "uniform", dtype)


__all__ = ["xavier_uniform"]
