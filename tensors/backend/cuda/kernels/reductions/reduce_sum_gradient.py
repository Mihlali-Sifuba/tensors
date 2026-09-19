"""CuPy implementation of the summation VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_sum_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Broadcast the upstream gradient back over the reduced axes."""
    upstream = _working_values(grad)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(value.shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = cupy.broadcast_to(expanded, value.shape).copy()
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
