"""NumPy implementation of slice scattering."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _shape_size
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def slice_scatter(
    value: Tensor, indices: list[int], *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Scatter flat values into a zero NumPy tensor."""
    working_dtype = (
        object if value.dtype.kind == "integer" else numpy.dtype(value.dtype.name)
    )
    result = numpy.zeros(_shape_size(output_shape), dtype=working_dtype)
    try:
        values = _view(value).reshape(-1).astype(working_dtype, copy=False)
    except ValueError:
        return None
    numpy.add.at(result, indices, values)
    return _storage(result, dtype=value.dtype, output_shape=output_shape)
