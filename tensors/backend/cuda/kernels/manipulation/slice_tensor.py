"""CuPy implementation of slicing."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import tensor_to_logical_array

if TYPE_CHECKING:
    from tensors._typing import TensorIndex
    from tensors.tensor import Tensor


def slice_tensor(
    value: Tensor, key: TensorIndex, *, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a NumPy slicing kernel after caller-side key validation."""
    try:
        result = tensor_to_logical_array(value)[key].copy()
    except ValueError:
        return None
    return _storage(result, dtype=value.dtype, output_shape=output_shape)
