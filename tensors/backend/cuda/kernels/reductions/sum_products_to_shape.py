"""CuPy implementation of a fused product summed to a broadcast shape."""

from __future__ import annotations
import cupy
import math
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _working_values
from tensors.backend.cuda.kernels.reductions.stability import _scaled_product_sum
from tensors.backend.cuda.kernels.reductions.stability import _stable_sum_candidate
from tensors.backend.cuda.kernels.reductions.stability import _sum_axes

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage | None:
    """Multiply and reduce broadcast VJP terms in one guarded kernel."""
    if gradient.dtype.kind == "integer":
        return None
    try:
        left, right = cupy.broadcast_arrays(
            _working_values(gradient),
            _working_values(factor),
        )
    except ValueError:
        return None
    axes = _sum_axes(tuple(left.shape), shape)
    if axes is None:
        return None
    finite = cupy.all(cupy.isfinite(left)) & cupy.all(cupy.isfinite(right))
    nonzero = (left != 0.0) & (right != 0.0)
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    with _errstate(divide="ignore", invalid="ignore"):
        log_terms = cupy.log(cupy.abs(left)) + cupy.log(cupy.abs(right))
    largest_log = cupy.max(
        cupy.where(nonzero, log_terms, -cupy.inf), axis=reduction_axes, keepdims=True
    )
    smallest_log = cupy.min(
        cupy.where(nonzero, log_terms, cupy.inf), axis=reduction_axes, keepdims=True
    )
    any_nonzero = cupy.any(nonzero, axis=reduction_axes, keepdims=True)
    comparable = ~any_nonzero | (
        largest_log - smallest_log <= -math.log(cupy.finfo(cupy.float64).eps)
    )
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        products = left * right
        direct = cupy.sum(products, axis=axes, keepdims=True) if axes else products
    lost_range = ~cupy.isfinite(products) | (products == 0.0) & nonzero
    direct_safe = (
        ~cupy.any(lost_range)
        & _stable_sum_candidate(products, axes)
        & ~cupy.any(cupy.isnan(direct))
    )
    stable = _scaled_product_sum(left, right, axes)
    result = cupy.where(direct_safe, direct, stable)
    valid = (
        finite & (direct_safe | cupy.all(comparable)) & ~cupy.any(cupy.isnan(result))
    )
    if not bool(valid):
        return None
    return _storage(
        cupy.asarray(result).reshape(shape), dtype=gradient.dtype, output_shape=shape
    )
