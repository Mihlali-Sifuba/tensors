"""NumPy implementation of axis permutation."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def transpose(
    value: Tensor, permutation: tuple[int, ...], *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Permute tensor axes into canonical contiguous storage."""
    result = numpy.transpose(_view(value), axes=permutation).copy()
    return _storage(result, dtype=value.dtype, output_shape=output_shape)
