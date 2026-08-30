"""Variance-scaling parameter initialization."""

from __future__ import annotations

import math

from ..random import normal, uniform
from ..random import _truncated_normal
from ..tensor import Tensor
from ..utils.shape import normalize_shape
from ._utils import (
    DType,
    Distribution,
    FanMode,
    Shape,
    calculate_fan_in_and_fan_out,
    finite_number,
    floating_dtype,
)


def variance_scaling(
    shape: Shape,
    scale: int | float = 1.0,
    mode: FanMode = "fan_in",
    distribution: Distribution = "truncated_normal",
    dtype: DType = None,
) -> Tensor:
    """Initialize a tensor with variance scale / fan."""
    normalized_shape = normalize_shape(shape)
    fan_in, fan_out = calculate_fan_in_and_fan_out(normalized_shape)
    factor = finite_number("scale", scale)
    if factor <= 0.0:
        raise ValueError("scale must be greater than zero")
    if mode == "fan_in":
        denominator = float(fan_in)
    elif mode == "fan_out":
        denominator = float(fan_out)
    elif mode == "fan_avg":
        denominator = (fan_in + fan_out) / 2.0
    else:
        raise ValueError("mode must be 'fan_in', 'fan_out', or 'fan_avg'")

    resolved_dtype = floating_dtype(dtype)
    variance = factor / denominator
    if distribution == "uniform":
        limit = math.sqrt(3.0 * variance)
        return uniform(normalized_shape, -limit, limit, dtype=resolved_dtype)
    if distribution == "normal":
        return normal(
            normalized_shape,
            stddev=math.sqrt(variance),
            dtype=resolved_dtype,
        )
    if distribution == "truncated_normal":
        correction = 0.8796256610342398
        source_stddev = math.sqrt(variance) / correction
        limit = 2.0 * source_stddev
        return _truncated_normal(
            normalized_shape,
            mean=0.0,
            stddev=source_stddev,
            lower=-limit,
            upper=limit,
            dtype=resolved_dtype,
        )
    raise ValueError(
        "distribution must be 'uniform', 'normal', or 'truncated_normal'"
    )


__all__ = ["variance_scaling"]
