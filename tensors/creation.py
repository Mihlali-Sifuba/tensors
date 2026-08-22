"""Public constructors for mathematically defined Tensor values."""

from __future__ import annotations

import math
from typing import Iterable, TypeAlias

from .dtype import DataType
from .tensor import Tensor
from .utils.shape import normalize_shape, shape_size


Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None
Scalar: TypeAlias = int | float


def full(shape: Shape, fill_value: Scalar, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with one constant value."""
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise TypeError("fill_value must be an int or float")
    normalized_shape = normalize_shape(shape)
    return Tensor(
        [fill_value] * shape_size(normalized_shape),
        dtype=dtype,
        shape=normalized_shape,
    )


def zeros(shape: Shape, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with zeros."""
    return full(shape, 0, dtype=dtype)


def ones(shape: Shape, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with ones."""
    return full(shape, 1, dtype=dtype)


def eye(
    rows: int,
    columns: int | None = None,
    k: int = 0,
    dtype: DType = None,
) -> Tensor:
    """Return a matrix with ones on diagonal ``k`` and zeros elsewhere."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    if columns is None:
        columns = rows
    if (
        isinstance(columns, bool)
        or not isinstance(columns, int)
        or columns < 0
    ):
        raise ValueError("columns must be a non-negative integer")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")

    values = [
        1 if column - row == k else 0
        for row in range(rows)
        for column in range(columns)
    ]
    return Tensor(values, dtype=dtype, shape=(rows, columns))


def arange(
    start: Scalar,
    stop: Scalar | None = None,
    step: Scalar = 1,
    dtype: DType = None,
) -> Tensor:
    """Return evenly spaced values in the half-open interval [start, stop)."""
    for name, value in (("start", start), ("stop", stop), ("step", step)):
        if value is None and name == "stop":
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an int or float")
    if step == 0:
        raise ValueError("step cannot be zero")
    if stop is None:
        start, stop = 0, start

    if all(isinstance(value, int) for value in (start, stop, step)):
        values = list(range(start, stop, step))
    else:
        try:
            finite = all(math.isfinite(value) for value in (start, stop, step))
        except OverflowError as exc:
            raise ValueError("arange values must be finite") from exc
        if not finite:
            raise ValueError("arange values must be finite")

        increasing = step > 0
        if (increasing and start >= stop) or (not increasing and start <= stop):
            values = []
        else:
            span = stop - start
            ratio = span / step
            if not math.isfinite(ratio):
                raise OverflowError("arange result is too large to construct")
            count = max(0, math.ceil(ratio))
            candidates = [start + index * step for index in range(count)]
            values = [
                value
                for value in candidates
                if (value < stop if increasing else value > stop)
            ]
    return Tensor(values, dtype=dtype, shape=(len(values),))


def linspace(
    start: Scalar,
    stop: Scalar,
    count: int = 50,
    dtype: DType = None,
) -> Tensor:
    """Return ``count`` evenly spaced values including both endpoints."""
    for name, value in (("start", start), ("stop", stop)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be an int or float")
        try:
            finite = math.isfinite(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not finite:
            raise ValueError(f"{name} must be finite")
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer")
    if count < 0:
        raise ValueError("count must be non-negative")

    if count == 0:
        values = []
    elif count == 1:
        values = [start]
    else:
        intervals = count - 1
        values = [start]
        for index in range(1, intervals):
            fraction = index / intervals
            values.append(math.fsum((
                float(start) * (1.0 - fraction),
                float(stop) * fraction,
            )))
        values.append(stop)
    return Tensor(values, dtype=dtype, shape=(count,))


__all__ = ["arange", "eye", "full", "linspace", "ones", "zeros"]
