"""Public constructors for mathematically defined Tensor values."""

from __future__ import annotations
import math
from typing import Iterable, TypeAlias
from tensors import dtype as _dtype
from tensors.backend import execute_arange, execute_eye, execute_full, execute_linspace
from tensors.dtype import DataType
from tensors.shape import Shape as TensorShape
from tensors.tensor import Tensor

Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None
Scalar: TypeAlias = int | float


def _resolve_dtype(dtype: DType) -> DataType:
    if dtype is None:
        return _dtype.default
    if isinstance(dtype, str):
        return _dtype.from_typecode(dtype)
    if not isinstance(dtype, DataType):
        raise TypeError(
            f"dtype must be a DataType, typecode or dtype string, got {type(dtype)}"
        )
    return dtype


def full(shape: Shape, fill_value: Scalar, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with one constant value."""
    if isinstance(fill_value, bool) or not isinstance(fill_value, (int, float)):
        raise TypeError("fill_value must be an int or float")
    normalized_shape = TensorShape.from_iterable(shape)
    resolved_dtype = _resolve_dtype(dtype)
    storage = execute_full(normalized_shape, fill_value, dtype=resolved_dtype)
    return Tensor._from_owned_storage(
        storage, dtype=resolved_dtype, shape=normalized_shape
    )


def zeros(shape: Shape, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with zeros."""
    return full(shape, 0, dtype=dtype)


def ones(shape: Shape, dtype: DType = None) -> Tensor:
    """Return a Tensor of ``shape`` filled with ones."""
    return full(shape, 1, dtype=dtype)


def eye(
    rows: int, columns: int | None = None, k: int = 0, dtype: DType = None
) -> Tensor:
    """Return a matrix with ones on diagonal ``k`` and zeros elsewhere."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a non-negative integer")
    if columns is None:
        columns = rows
    if isinstance(columns, bool) or not isinstance(columns, int) or columns < 0:
        raise ValueError("columns must be a non-negative integer")
    if isinstance(k, bool) or not isinstance(k, int):
        raise TypeError("k must be an integer")
    resolved_dtype = _resolve_dtype(dtype)
    storage = execute_eye(rows, columns, k, dtype=resolved_dtype)
    return Tensor._from_owned_storage(
        storage, dtype=resolved_dtype, shape=(rows, columns)
    )


def arange(
    start: Scalar, stop: Scalar | None = None, step: Scalar = 1, dtype: DType = None
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
        start, stop = (0, start)
    resolved_dtype = _resolve_dtype(dtype)
    if isinstance(start, int) and isinstance(stop, int) and isinstance(step, int):
        count = len(range(start, stop, step))
    else:
        try:
            finite = all((math.isfinite(value) for value in (start, stop, step)))
        except OverflowError as exc:
            raise ValueError("arange values must be finite") from exc
        if not finite:
            raise ValueError("arange values must be finite")
        increasing = step > 0
        if increasing and start >= stop or (not increasing and start <= stop):
            count = 0
        else:
            span = stop - start
            ratio = span / step
            if not math.isfinite(ratio):
                raise OverflowError("arange result is too large to construct")
            count = max(0, math.ceil(ratio))
            # Rounding can place the last candidate on or past ``stop``; the
            # interval is half-open, so drop it rather than emit it.
            while count:
                final = start + (count - 1) * step
                if final < stop if increasing else final > stop:
                    break
                count -= 1
    storage = execute_arange(start, step, count, dtype=resolved_dtype)
    return Tensor._from_owned_storage(storage, dtype=resolved_dtype, shape=(count,))


def linspace(
    start: Scalar, stop: Scalar, count: int = 50, dtype: DType = None
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
    resolved_dtype = _resolve_dtype(dtype)
    accelerated = execute_linspace(start, stop, count, dtype=resolved_dtype)
    return Tensor._from_owned_storage(accelerated, dtype=resolved_dtype, shape=(count,))


__all__ = ["arange", "eye", "full", "linspace", "ones", "zeros"]
