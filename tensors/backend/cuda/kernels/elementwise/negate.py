"""CuPy implementation of negation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def negate(value: Tensor, *, dtype: DataType) -> Storage | None:
    """Run elementwise NumPy negation."""
    try:
        operand = _operand(value, dtype)
    except (TypeError, ValueError):
        return None
    result = cupy.negative(operand)
    return _storage(result, dtype=dtype, output_shape=value.shape)
