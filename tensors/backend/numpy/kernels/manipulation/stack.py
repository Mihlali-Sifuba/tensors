"""NumPy implementation of stacking along a new axis."""

from __future__ import annotations
import numpy
from collections.abc import Sequence
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def stack(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Stack tensors along a newly inserted axis."""
    try:
        result = numpy.stack(
            [tensor_to_logical_array(value) for value in values], axis=axis
        )
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
