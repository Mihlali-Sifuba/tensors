"""CUDA implementation of the product."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.conversion import _arithmetic_storage as _storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _widen
from tensors.backend.cuda.kernels.reductions.exact import exact_integer_product
from tensors.backend.storage import Storage

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
    """Run a native product with exact integer overflow semantics."""
    if values.size == 0:
        return _storage(
            cupy.full(output_shape, 1, dtype=cupy.dtype(dtype.name)),
            dtype=dtype,
            output_shape=output_shape,
        )
    if dtype.kind == "integer":
        result = exact_integer_product(
            values,
            input_shape,
            axes,
            dtype=dtype,
            output_shape=output_shape,
        )
    else:
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = cupy.prod(
                _widen(values),
                axis=axes,
                keepdims=keepdims,
            )
    return _storage(result, dtype=dtype, output_shape=output_shape)
