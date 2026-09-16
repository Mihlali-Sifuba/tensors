"""NumPy implementation of the matrix product."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor
from tensors.backend.numpy.kernels.linalg.matmul_ops import _comparable_finite_values
from tensors.backend.numpy.kernels.linalg.matmul_ops import _scaled_matmul


def matmul(
    left: Tensor, right: Tensor, *, dtype: DataType, output_shape: tuple[int, ...]
) -> Storage | None:
    """Return a NumPy matrix product or defer to the reference implementation."""
    if dtype.kind != "floating":
        return None
    try:
        left_array = _view(left).astype(numpy.float64, copy=False)
        right_array = _view(right).astype(numpy.float64, copy=False)
    except ValueError:
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        result = numpy.matmul(left_array, right_array)
    if not bool(numpy.all(numpy.isfinite(result))):
        if not (
            _comparable_finite_values(left_array)
            and _comparable_finite_values(right_array)
        ):
            return None
        result = _scaled_matmul(left_array, right_array)
        if bool(numpy.any(numpy.isnan(result))):
            return None
    return _storage(result, dtype=dtype, output_shape=output_shape)
