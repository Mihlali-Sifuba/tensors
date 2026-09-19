"""CuPy implementation of the product."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def reduce_prod(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a numerically guarded NumPy reduction."""
    if value.size == 0:
        return None
    axis = axes
    values = _view(value)
    working = values.astype(object) if dtype.kind == "integer" else _widen(values)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = cupy.prod(working, axis=axis, keepdims=keepdims)
    return _storage(result, dtype=dtype, output_shape=output_shape)
