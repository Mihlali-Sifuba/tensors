"""Structural kernels: shape, layout, indexing, and representation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ..storage import Storage
from .core import _cuda_integer, _numpy, _shape_size, _storage, _view

if TYPE_CHECKING:
    from ..._typing import TensorIndex
    from ...dtype import DataType
    from ...tensor import Tensor

def slice_tensor(
    value: Tensor,
    key: TensorIndex,
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Run a NumPy slicing kernel after caller-side key validation."""
    numpy = _numpy()
    try:
        # Provider slicing can return a view into the input Tensor. Backend
        # results transferred into a new Tensor must own independent storage.
        result = _view(value, numpy)[key].copy()
    except ValueError:
        return None
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def slice_scatter(
    value: Tensor,
    indices: list[int],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Scatter flat values into a zero NumPy tensor."""
    if _cuda_integer(value.dtype):
        return None
    numpy = _numpy()
    working_dtype = object if value.dtype.kind == "integer" else numpy.dtype(
        value.dtype.name
    )
    result = numpy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        values = _view(value, numpy).reshape(-1).astype(
            working_dtype,
            copy=False,
        )
    except ValueError:
        return None
    numpy.add.at(result, indices, values)
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def cast_tensor(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Convert tensor values with Python-compatible scalar conversion."""
    if _cuda_integer(dtype):
        return None
    numpy = _numpy()
    try:
        source = _view(value, numpy).reshape(-1)
    except ValueError:
        return None
    if dtype.kind == "integer":
        converter = numpy.frompyfunc(int, 1, 1)
        result = converter(source)
    else:
        # Same-dtype casts would otherwise retain the source buffer.
        result = source.astype(numpy.float64, copy=True)
    return _storage(
        result,
        dtype=dtype,
        output_shape=value.shape,
        numpy=numpy,
    )

def transpose(
    value: Tensor,
    permutation: tuple[int, ...],
    *,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Permute tensor axes into canonical contiguous storage."""
    numpy = _numpy()
    # Transpose is a provider view operation, while the public Tensor result is
    # materialized and independently owned.
    result = numpy.transpose(
        _view(value, numpy),
        axes=permutation,
    ).copy()
    return _storage(
        result,
        dtype=value.dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Concatenate tensors along an existing axis."""
    numpy = _numpy()
    try:
        result = numpy.concatenate(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )

def stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Stack tensors along a newly inserted axis."""
    numpy = _numpy()
    try:
        result = numpy.stack(
            [_view(value, numpy) for value in values],
            axis=axis,
        )
    except (TypeError, ValueError):
        return None
    return _storage(
        result,
        dtype=dtype,
        output_shape=output_shape,
        numpy=numpy,
    )
