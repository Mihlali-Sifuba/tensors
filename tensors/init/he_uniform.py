"""He uniform parameter initialization."""

from __future__ import annotations

from dataclasses import dataclass

from ..tensor import Tensor
from ._utils import DType, Shape, floating_dtype
from .initializer import Initializer
from .variance_scaling import VarianceScaling


@dataclass(frozen=True, slots=True)
class HeUniform(Initializer):
    """Reusable He uniform initializer configuration."""

    dtype: DType = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", floating_dtype(self.dtype))

    def __call__(self, shape: Shape) -> Tensor:
        return VarianceScaling(2.0, "fan_in", "uniform", self.dtype)(shape)


def he_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with uniform variance 2 / fan_in."""
    return HeUniform(dtype)(shape)


__all__ = ["HeUniform", "he_uniform"]
