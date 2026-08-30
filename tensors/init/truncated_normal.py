"""Truncated-normal parameter initialization."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

from ..random import _truncated_normal
from ..tensor import Tensor
from ._utils import DType, Shape, finite_number, floating_dtype
from .initializer import Initializer


@dataclass(frozen=True, slots=True)
class TruncatedNormal(Initializer):
    """Reusable truncated-normal initializer configuration."""

    mean: int | float = 0.0
    stddev: int | float = 1.0
    lower: int | float | None = None
    upper: int | float | None = None
    dtype: DType = None

    def __post_init__(self) -> None:
        center = finite_number("mean", self.mean)
        deviation = finite_number("stddev", self.stddev)
        if deviation <= 0.0:
            raise ValueError("stddev must be greater than zero")
        minimum = (
            center - 2.0 * deviation
            if self.lower is None
            else finite_number("lower", self.lower)
        )
        maximum = (
            center + 2.0 * deviation
            if self.upper is None
            else finite_number("upper", self.upper)
        )
        if not minimum < maximum:
            raise ValueError("lower must be less than upper")
        distribution = NormalDist(center, deviation)
        retained = distribution.cdf(maximum) - distribution.cdf(minimum)
        if retained < 1e-6:
            raise ValueError(
                "truncation bounds retain too little probability to sample "
                "safely"
            )
        object.__setattr__(self, "mean", center)
        object.__setattr__(self, "stddev", deviation)
        object.__setattr__(self, "lower", minimum)
        object.__setattr__(self, "upper", maximum)
        object.__setattr__(self, "dtype", floating_dtype(self.dtype))

    def __call__(self, shape: Shape) -> Tensor:
        """Sample the configured normal distribution within its bounds."""
        assert self.lower is not None
        assert self.upper is not None
        return _truncated_normal(
            shape,
            mean=self.mean,
            stddev=self.stddev,
            lower=self.lower,
            upper=self.upper,
            dtype=self.dtype,
        )


def truncated_normal(
    shape: Shape,
    mean: int | float = 0.0,
    stddev: int | float = 1.0,
    lower: int | float | None = None,
    upper: int | float | None = None,
    dtype: DType = None,
) -> Tensor:
    """Sample using a one-shot TruncatedNormal configuration."""
    return TruncatedNormal(mean, stddev, lower, upper, dtype)(shape)


__all__ = ["TruncatedNormal", "truncated_normal"]
