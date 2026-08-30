"""Truncated-normal parameter initialization."""

from __future__ import annotations

from ..random import _truncated_normal
from ..tensor import Tensor
from ._utils import DType, Shape, finite_number


def truncated_normal(
    shape: Shape,
    mean: int | float = 0.0,
    stddev: int | float = 1.0,
    lower: int | float | None = None,
    upper: int | float | None = None,
    dtype: DType = None,
) -> Tensor:
    """Sample a normal distribution subject to inclusive absolute bounds.

    stddev describes the source normal before truncation. Omitted bounds
    default to two source standard deviations on either side of mean.
    """
    center = finite_number("mean", mean)
    deviation = finite_number("stddev", stddev)
    if deviation <= 0.0:
        raise ValueError("stddev must be greater than zero")
    minimum = center - 2.0 * deviation if lower is None else lower
    maximum = center + 2.0 * deviation if upper is None else upper
    return _truncated_normal(
        shape,
        mean=center,
        stddev=deviation,
        lower=minimum,
        upper=maximum,
        dtype=dtype,
    )


__all__ = ["truncated_normal"]
