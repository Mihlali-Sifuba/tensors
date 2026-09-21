"""CuPy implementation of axis permutation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def transpose(
    value: Tensor, permutation: tuple[int, ...], *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Permute tensor axes into canonical contiguous storage."""
    result = cupy.transpose(tensor_to_logical_array(value), axes=permutation).copy()
    return _storage(result, dtype=value.dtype, output_shape=output_shape)
