"""Fan calculation and variance-scaling parameter initializers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, TypeAlias

from ..creation import _resolve_dtype
from ..dtype import DataType
from ..random import normal, uniform
from ..random import _truncated_normal
from ..tensor import Tensor
from ..utils.shape import normalize_shape


Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None
FanMode: TypeAlias = Literal["fan_in", "fan_out", "fan_avg"]
Distribution: TypeAlias = Literal["uniform", "normal", "truncated_normal"]


def _floating_dtype(dtype: DType) -> DataType:
    resolved = _resolve_dtype(dtype)
    if resolved.kind != "floating":
        raise TypeError("parameter initializers require a floating dtype")
    return resolved


def _finite_number(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def _calculate_fan_in_and_fan_out(shape: Shape) -> tuple[int, int]:
    """Return channel-last fan values for a weight tensor.

    Matrices use (fan_in, fan_out). For rank greater than two, leading
    dimensions form the receptive field and the final two dimensions are the
    input and output channel counts: (..., in_channels, out_channels).
    """
    normalized = normalize_shape(shape)
    if len(normalized) < 2:
        raise ValueError(
            "fan_in and fan_out require a shape with at least two dimensions"
        )
    if any(dimension == 0 for dimension in normalized):
        raise ValueError(
            "fan_in and fan_out require all dimensions to be positive"
        )
    receptive_field = math.prod(normalized[:-2])
    return (
        receptive_field * normalized[-2],
        receptive_field * normalized[-1],
    )


def variance_scaling(
    shape: Shape,
    scale: int | float = 1.0,
    mode: FanMode = "fan_in",
    distribution: Distribution = "truncated_normal",
    dtype: DType = None,
) -> Tensor:
    """Initialize a tensor with variance scale / fan.

    fan is fan_in, fan_out, or their arithmetic mean according to mode.
    Uniform bounds are sqrt(3 * scale / fan). Normal standard deviation is
    sqrt(scale / fan). Truncated normal corrects for truncation at two source
    standard deviations so the retained values have the requested variance.
    """
    normalized_shape = normalize_shape(shape)
    fan_in, fan_out = _calculate_fan_in_and_fan_out(normalized_shape)
    factor = _finite_number("scale", scale)
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
    resolved_dtype = _floating_dtype(dtype)
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


def xavier_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """Xavier uniform initialization with variance 2 / (fan_in + fan_out)."""
    return variance_scaling(shape, 1.0, "fan_avg", "uniform", dtype)


def xavier_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """Xavier normal initialization with variance 2 / (fan_in + fan_out)."""
    return variance_scaling(shape, 1.0, "fan_avg", "normal", dtype)


def he_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """He uniform initialization with variance 2 / fan_in."""
    return variance_scaling(shape, 2.0, "fan_in", "uniform", dtype)


def he_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """He normal initialization with variance 2 / fan_in."""
    return variance_scaling(shape, 2.0, "fan_in", "normal", dtype)


def lecun_uniform(shape: Shape, dtype: DType = None) -> Tensor:
    """LeCun uniform initialization with variance 1 / fan_in."""
    return variance_scaling(shape, 1.0, "fan_in", "uniform", dtype)


def lecun_normal(shape: Shape, dtype: DType = None) -> Tensor:
    """LeCun normal initialization with variance 1 / fan_in."""
    return variance_scaling(shape, 1.0, "fan_in", "normal", dtype)


__all__ = [
    "he_normal",
    "he_uniform",
    "lecun_normal",
    "lecun_uniform",
    "variance_scaling",
    "xavier_normal",
    "xavier_uniform",
]
