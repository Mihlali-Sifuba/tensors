"""Kernels that build new arrays from parameters rather than transform one."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..storage import Storage
from .core import _cuda_integer, _errstate, _is_cuda, _numpy, _storage, _view

if TYPE_CHECKING:
    from ...dtype import DataType
    from ...tensor import Tensor

def one_hot_targets(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> Storage | None:
    """Expand validated class indices directly into native dense storage."""
    numpy = _numpy()
    values = _view(targets, numpy).astype(numpy.float64, copy=False)
    class_count = logits.shape[axis]
    with _errstate(numpy, invalid="ignore"):
        integral = values == numpy.floor(values)
    valid = numpy.all(
        numpy.isfinite(values)
        & integral
        & (values >= 0.0)
        & (values < class_count)
    )
    if not bool(valid):
        return None

    sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
    indices = values.astype(numpy.int64).reshape(sample_shape)
    expanded_indices = numpy.expand_dims(indices, axis=axis)
    result = numpy.zeros(logits.shape, dtype=numpy.float64)
    numpy.put_along_axis(result, expanded_indices, 1.0, axis=axis)
    return _storage(
        result,
        dtype=logits.dtype,
        output_shape=logits.shape,
        numpy=numpy,
    )

def full(
    shape: tuple[int, ...],
    fill_value: int | float,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create constant-filled canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.full(shape, fill_value, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=shape,
        numpy=numpy,
    )

def eye(
    rows: int,
    columns: int,
    k: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create identity-like canonical storage."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    working_dtype = object if dtype.kind == "integer" else numpy.float64
    result = numpy.eye(rows, columns, k=k, dtype=working_dtype)
    return _storage(
        result,
        dtype=dtype,
        output_shape=(rows, columns),
        numpy=numpy,
    )

def arange(
    start: int | float,
    step: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create an arithmetic progression from a validated element count."""
    integer_inputs = isinstance(start, int) and isinstance(step, int)
    if _cuda_integer(dtype) or (_is_cuda() and integer_inputs):
        return None
    numpy = _numpy()
    working_dtype = object if integer_inputs else numpy.float64
    indices = numpy.arange(count, dtype=working_dtype)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start + indices * step
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )

def linspace(
    start: int | float,
    stop: int | float,
    count: int,
    *,
    dtype: DataType,
) -> Storage | None:
    """Create evenly spaced values when ordinary vector arithmetic is safe."""
    if count < 2:
        return None
    start_value = float(start)
    stop_value = float(stop)
    if start_value * stop_value < 0.0:
        return None
    numpy = _numpy()
    limit = numpy.finfo(numpy.float64).max / 4.0
    if abs(start_value) > limit or abs(stop_value) > limit:
        return None
    fractions = numpy.arange(count, dtype=numpy.float64) / (count - 1)
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        result = start_value * (1.0 - fractions) + stop_value * fractions
    result[0] = start
    result[-1] = stop
    return _storage(
        result,
        dtype=dtype,
        output_shape=(count,),
        numpy=numpy,
    )
