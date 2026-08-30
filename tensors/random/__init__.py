"""Backend-native random tensor generation.

This module owns independent Python, NumPy, and CUDA generator streams.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from statistics import NormalDist
from typing import TypeAlias

from .. import dtype as _dtype
from ..creation import _resolve_dtype
from ..dtype import DataType
from ..shape import Shape as TensorShape
from ..tensor import Tensor
from . import _state


Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None


def _floating_dtype(dtype: DType) -> DataType:
    resolved = _resolve_dtype(dtype)
    if resolved.kind != "floating":
        raise TypeError("random floating-point tensors require a floating dtype")
    return resolved


def _finite_number(name: str, value: int | float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def seed(value: int | None) -> None:
    """Reset all backend RNG streams.

    Integer seeds are limited to [0, 2**32) so every provider accepts them.
    None restores entropy-based initialization. Streams are reproducible
    independently, but different providers need not produce identical values.
    This function does not modify global Python, NumPy, or CuPy RNG state.
    """
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("seed must be an integer or None")
        if not 0 <= value < 2**32:
            raise ValueError("seed must be in the range [0, 2**32)")
    _state.seed(value)


def uniform(
    shape: Shape,
    low: int | float = 0.0,
    high: int | float = 1.0,
    dtype: DType = None,
) -> Tensor:
    """Return samples from the continuous uniform distribution [low, high)."""
    normalized_shape = TensorShape.from_iterable(shape)
    resolved_dtype = _floating_dtype(dtype)
    lower = _finite_number("low", low)
    upper = _finite_number("high", high)
    if not lower < upper:
        raise ValueError("low must be less than high")
    storage = _state.uniform(
        normalized_shape.size, lower, upper, resolved_dtype
    )
    return Tensor(storage, dtype=resolved_dtype, shape=normalized_shape)


def normal(
    shape: Shape,
    mean: int | float = 0.0,
    stddev: int | float = 1.0,
    dtype: DType = None,
) -> Tensor:
    """Return samples from N(mean, stddev**2)."""
    normalized_shape = TensorShape.from_iterable(shape)
    resolved_dtype = _floating_dtype(dtype)
    center = _finite_number("mean", mean)
    deviation = _finite_number("stddev", stddev)
    if deviation < 0.0:
        raise ValueError("stddev must be non-negative")
    storage = _state.normal(
        normalized_shape.size, center, deviation, resolved_dtype
    )
    return Tensor(storage, dtype=resolved_dtype, shape=normalized_shape)


def _integer_bounds(dtype: DataType) -> tuple[int, int]:
    bits = dtype.size * 8
    if dtype.typecode == _dtype.uint8.typecode:
        return 0, 2**bits - 1
    return -(2 ** (bits - 1)), 2 ** (bits - 1) - 1


def randint(
    shape: Shape,
    low: int,
    high: int | None = None,
    dtype: DType = _dtype.int64,
) -> Tensor:
    """Return integers sampled uniformly from a half-open interval.

    If high is omitted, values are drawn from [0, low).
    """
    normalized_shape = TensorShape.from_iterable(shape)
    resolved_dtype = _resolve_dtype(dtype)
    if resolved_dtype.kind != "integer":
        raise TypeError("randint requires an integer dtype")
    for name, value in (("low", low), ("high", high)):
        if value is None and name == "high":
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if high is None:
        low, high = 0, low
    if low >= high:
        raise ValueError("low must be less than high")
    minimum, maximum = _integer_bounds(resolved_dtype)
    if low < minimum or high - 1 > maximum:
        raise ValueError(
            f"requested interval is outside the range of {resolved_dtype.name}"
        )
    storage = _state.randint(
        normalized_shape.size, low, high, resolved_dtype
    )
    return Tensor(storage, dtype=resolved_dtype, shape=normalized_shape)


def _truncated_normal(
    shape: Shape,
    *,
    mean: int | float,
    stddev: int | float,
    lower: int | float,
    upper: int | float,
    dtype: DType,
) -> Tensor:
    """Internal backend-native truncated-normal constructor."""
    normalized_shape = TensorShape.from_iterable(shape)
    resolved_dtype = _floating_dtype(dtype)
    center = _finite_number("mean", mean)
    deviation = _finite_number("stddev", stddev)
    minimum = _finite_number("lower", lower)
    maximum = _finite_number("upper", upper)
    if deviation <= 0.0:
        raise ValueError("stddev must be greater than zero")
    if not minimum < maximum:
        raise ValueError("lower must be less than upper")
    distribution = NormalDist(center, deviation)
    retained_probability = (
        distribution.cdf(maximum) - distribution.cdf(minimum)
    )
    if retained_probability < 1e-6:
        raise ValueError(
            "truncation bounds retain too little probability to sample safely"
        )
    storage = _state.truncated_normal(
        normalized_shape.size,
        center,
        deviation,
        minimum,
        maximum,
        resolved_dtype,
    )
    return Tensor(storage, dtype=resolved_dtype, shape=normalized_shape)


__all__ = ["normal", "randint", "seed", "uniform"]
