"""CuPy implementation of the product VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _storage
from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def reduce_prod_gradient(
    grad: Tensor, value: Tensor, axes: tuple[int, ...], *, keepdims: bool
) -> Storage | None:
    """Run fused VJPs for reductions with regular native fast paths."""
    values = _view(value).astype(cupy.float64, copy=False)
    upstream = _view(grad).astype(cupy.float64, copy=False)
    expanded_shape = tuple(
        (1 if dimension in axes else size for dimension, size in enumerate(value.shape))
    )
    try:
        expanded = upstream.reshape(expanded_shape)
    except ValueError:
        return None
    count = 1
    for axis in axes:
        count *= value.shape[axis]
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if not bool(cupy.all(cupy.isfinite(values))):
            return None
        zero_count = cupy.sum(values == 0.0, axis=axes, keepdims=True)
        product = cupy.prod(values, axis=axes, keepdims=True)
        nonzero_product = cupy.prod(
            cupy.where(values == 0.0, 1.0, values), axis=axes, keepdims=True
        )
        unsafe = (zero_count == 0) & ((product == 0.0) | ~cupy.isfinite(product)) | (
            zero_count == 1
        ) & ((nonzero_product == 0.0) | ~cupy.isfinite(nonzero_product))
        if bool(cupy.any(unsafe)):
            return None
        derivative = cupy.where(
            zero_count == 0,
            product / values,
            cupy.where((zero_count == 1) & (values == 0.0), nonzero_product, 0.0),
        )
        result = expanded * derivative
    return _storage(result, dtype=grad.dtype, output_shape=value.shape)
