"""Xavier uniform parameter initialization."""

from __future__ import annotations

from dataclasses import dataclass

from ..tensor import Tensor
from ._utils import DType, Shape, floating_dtype
from .initializer import Initializer
from .variance_scaling import VarianceScaling


@dataclass(frozen=True, slots=True)
class XavierUniform(Initializer):
    """Reusable Xavier uniform initializer configuration."""

    dtype: DType = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", floating_dtype(self.dtype))

    def __call__(self, shape: Shape) -> Tensor:
        return VarianceScaling(1.0, "fan_avg", "uniform", self.dtype)(shape)


def xavier_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with uniform variance 2 / (fan_in + fan_out)."""
    return XavierUniform(dtype)(shape)


__all__ = ["XavierUniform", "xavier_uniform"]
