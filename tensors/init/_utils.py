"""Shared validation and fan helpers for parameter initializers."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Literal, TypeAlias

from ..creation import _resolve_dtype
from ..dtype import DataType
from ..shape import Shape as TensorShape


Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None
FanMode: TypeAlias = Literal["fan_in", "fan_out", "fan_avg"]
Distribution: TypeAlias = Literal["uniform", "normal", "truncated_normal"]


def floating_dtype(dtype: DType) -> DataType:
    """Resolve and validate a floating initializer dtype."""
    resolved = _resolve_dtype(dtype)
    if resolved.kind != "floating":
        raise TypeError("parameter initializers require a floating dtype")
    return resolved


def finite_number(name: str, value: int | float) -> float:
    """Return a finite float after validating one numeric argument."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be an int or float")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite")
    return converted


def calculate_fan_in_and_fan_out(shape: Shape) -> tuple[int, int]:
    """Return channel-last fan values for a weight tensor.

    Matrices use (fan_in, fan_out). For rank greater than two, leading
    dimensions form the receptive field and the final dimensions are
    (in_channels, out_channels).
    """
    normalized = TensorShape.from_iterable(shape)
    if len(normalized) < 2:
        raise ValueError(
            "fan_in and fan_out require a shape with at least two dimensions"
        )
    if any(dimension == 0 for dimension in normalized):
        raise ValueError(
            "fan_in and fan_out require all dimensions to be positive"
        )
    receptive_field = normalized[:-2].size
    return (
        receptive_field * normalized[-2],
        receptive_field * normalized[-1],
    )


__all__ = [
    "DType",
    "Distribution",
    "FanMode",
    "Shape",
    "calculate_fan_in_and_fan_out",
    "finite_number",
    "floating_dtype",
]
