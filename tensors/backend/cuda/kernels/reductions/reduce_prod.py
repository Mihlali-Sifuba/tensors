"""CuPy implementation of the product."""

from __future__ import annotations
import cupy
from typing import Any, TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def reduce_prod(
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if values.size == 0:
        return _storage(
            cupy.full(output_shape, 1), dtype=dtype, output_shape=output_shape
        )
    axis = axes
    working = values if dtype.kind == "integer" else _widen(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = cupy.prod(working, axis=axis, keepdims=keepdims)
    return _storage(result, dtype=dtype, output_shape=output_shape)
