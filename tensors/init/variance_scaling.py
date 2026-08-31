"""Variance-scaling parameter initialization."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..random import normal, uniform
from ..random import _truncated_normal
from ..shape import Shape as TensorShape
from ..tensor import Tensor
from ._utils import (
    DType,
    Distribution,
    FanMode,
    Shape,
    calculate_fan_in_and_fan_out,
    finite_number,
    floating_dtype,
)
from .initializer import Initializer


@dataclass(frozen=True, slots=True)
class VarianceScaling(Initializer):
    """Reusable variance-scaling initializer configuration."""

    scale: int | float = 1.0
    mode: FanMode = "fan_in"
    distribution: Distribution = "truncated_normal"
    dtype: DType = None

    def __post_init__(self) -> None:
        factor = finite_number("scale", self.scale)
        if factor <= 0.0:
            raise ValueError("scale must be greater than zero")
        if self.mode not in {"fan_in", "fan_out", "fan_avg"}:
            raise ValueError(
                "mode must be 'fan_in', 'fan_out', or 'fan_avg'"
            )
        if self.distribution not in {
            "uniform",
            "normal",
            "truncated_normal",
        }:
            raise ValueError(
                "distribution must be 'uniform', 'normal', or "
                "'truncated_normal'"
            )
        object.__setattr__(self, "scale", factor)
        object.__setattr__(self, "dtype", floating_dtype(self.dtype))

    def __call__(self, shape: Shape) -> Tensor:
        """Initialize a tensor with variance scale / fan."""
        normalized_shape = TensorShape.from_iterable(shape)
        fan_in, fan_out = calculate_fan_in_and_fan_out(normalized_shape)
        if self.mode == "fan_in":
            denominator = float(fan_in)
        elif self.mode == "fan_out":
            denominator = float(fan_out)
        else:
            denominator = (fan_in + fan_out) / 2.0

        variance = float(self.scale) / denominator
        if self.distribution == "uniform":
            limit = math.sqrt(3.0 * variance)
            return uniform(
                normalized_shape,
                -limit,
                limit,
                dtype=self.dtype,
            )
        if self.distribution == "normal":
            return normal(
                normalized_shape,
                stddev=math.sqrt(variance),
                dtype=self.dtype,
            )
        correction = 0.8796256610342398
        source_stddev = math.sqrt(variance) / correction
        limit = 2.0 * source_stddev
        return _truncated_normal(
            normalized_shape,
            mean=0.0,
            stddev=source_stddev,
            lower=-limit,
            upper=limit,
            dtype=self.dtype,
        )


def variance_scaling(
    shape: Shape,
    scale: int | float = 1.0,
    mode: FanMode = "fan_in",
    distribution: Distribution = "truncated_normal",
    dtype: DType = None,
) -> Tensor:
    """Initialize a tensor using a one-shot VarianceScaling configuration."""
    return VarianceScaling(scale, mode, distribution, dtype)(shape)


__all__ = ["VarianceScaling", "variance_scaling"]
