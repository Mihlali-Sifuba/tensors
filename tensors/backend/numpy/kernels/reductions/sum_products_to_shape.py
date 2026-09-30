"""NumPy implementation of a product summed to a broadcast shape."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _storage
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.kernels.reductions.pairwise import (
    pairwise_float_sum,
    to_declared_dtype,
)
from tensors.backend.numpy.kernels.reductions.stability import _sum_axes

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage | None:
    """Multiply broadcast VJP terms, then sum them to ``shape`` pairwise.

    Each product is formed once in the gradient's dtype and the products that
    reduce to one target position are summed in logical order with the
    package's pairwise tree, as every contraction is.
    """
    if gradient.dtype.kind == "integer":
        return None
    try:
        left, right = numpy.broadcast_arrays(
            to_declared_dtype(tensor_to_logical_array(gradient), gradient.dtype),
            to_declared_dtype(tensor_to_logical_array(factor), gradient.dtype),
        )
    except ValueError:
        return None
    axes = _sum_axes(tuple(left.shape), shape)
    if axes is None:
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        products = left * right
    result = pairwise_float_sum(products, axes)
    return _storage(
        numpy.asarray(result).reshape(shape), dtype=gradient.dtype, output_shape=shape
    )
