"""LeCun normal parameter initialization."""

from __future__ import annotations

from dataclasses import dataclass

from ..tensor import Tensor
from ._utils import DType, Shape, floating_dtype
from .initializer import Initializer
from .variance_scaling import VarianceScaling


@dataclass(frozen=True, slots=True)
class LecunNormal(Initializer):
    """Reusable LeCun normal initializer configuration."""

    dtype: DType = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", floating_dtype(self.dtype))

    def __call__(self, shape: Shape) -> Tensor:
        return VarianceScaling(1.0, "fan_in", "normal", self.dtype)(shape)


def lecun_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """Initialize with normal variance 1 / fan_in."""
    return LecunNormal(dtype)(shape)


__all__ = ["LecunNormal", "lecun_normal"]
