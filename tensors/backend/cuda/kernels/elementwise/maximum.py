"""CuPy implementation of the elementwise maximum."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def maximum(
    left: Tensor, right: Tensor, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage | None:
    """Run a broadcasting elementwise minimum or maximum."""
    function = cupy.maximum
    try:
        # Widened first: CuPy's binary32 elementwise code flushes a
        # subnormal operand, so the comparison would be made between
        # values the caller did not supply. In binary64 it is not,
        # and _storage narrows the result back through PTX.
        result = function(_working_values(left), _working_values(right))
    except (TypeError, ValueError):
        return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
