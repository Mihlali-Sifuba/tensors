"""Summation of values and products down to a broadcast shape."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from ...storage import Storage
from ..core import _errstate, _numpy, _storage, _view
from .stability import (
    _scaled_product_sum,
    _scaled_sum,
    _stable_sum_candidate,
    _sum_axes,
)

if TYPE_CHECKING:
    from ....tensor import Tensor

def sum_to_shape(
    gradient: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Reduce a broadcast gradient using guarded native summation."""
    if gradient.dtype.kind == "integer":
        return None
    layout = _sum_axes(gradient.shape, shape)
    if layout is None:
        return None
    _, axes = layout
    numpy = _numpy()
    values = _view(gradient, numpy).astype(numpy.float64, copy=False)
    safe = _stable_sum_candidate(values, axes, numpy)
    result = _scaled_sum(values, axes, numpy)
    valid = safe & ~numpy.any(numpy.isnan(result))
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )

def sum_products_to_shape(
    gradient: Tensor,
    factor: Tensor,
    shape: tuple[int, ...],
) -> Storage | None:
    """Multiply and reduce broadcast VJP terms in one guarded kernel."""
    if gradient.dtype.kind == "integer":
        return None
    numpy = _numpy()
    try:
        left, right = numpy.broadcast_arrays(
            _view(gradient, numpy).astype(numpy.float64, copy=False),
            _view(factor, numpy).astype(numpy.float64, copy=False),
        )
    except ValueError:
        return None
    layout = _sum_axes(tuple(left.shape), shape)
    if layout is None:
        return None
    _, axes = layout
    finite = numpy.all(numpy.isfinite(left)) & numpy.all(numpy.isfinite(right))
    nonzero = (left != 0.0) & (right != 0.0)
    reduction_axes: tuple[int, ...] | None = axes if axes else None
    with _errstate(numpy, divide="ignore", invalid="ignore"):
        log_terms = numpy.log(numpy.abs(left)) + numpy.log(numpy.abs(right))
    largest_log = numpy.max(
        numpy.where(nonzero, log_terms, -numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    smallest_log = numpy.min(
        numpy.where(nonzero, log_terms, numpy.inf),
        axis=reduction_axes,
        keepdims=True,
    )
    any_nonzero = numpy.any(nonzero, axis=reduction_axes, keepdims=True)
    comparable = (~any_nonzero) | (
        largest_log - smallest_log
        <= -math.log(numpy.finfo(numpy.float64).eps)
    )
    with _errstate(numpy, over="ignore", under="ignore", invalid="ignore"):
        products = left * right
        direct = (
            numpy.sum(products, axis=axes, keepdims=True)
            if axes
            else products
        )
    lost_range = (~numpy.isfinite(products)) | (
        (products == 0.0) & nonzero
    )
    direct_safe = (
        ~numpy.any(lost_range)
        & _stable_sum_candidate(products, axes, numpy)
        & ~numpy.any(numpy.isnan(direct))
    )
    stable = _scaled_product_sum(left, right, axes, numpy)
    result = numpy.where(direct_safe, direct, stable)
    valid = (
        finite
        & (direct_safe | numpy.all(comparable))
        & ~numpy.any(numpy.isnan(result))
    )
    if not bool(valid):
        return None
    return _storage(
        numpy.asarray(result).reshape(shape),
        dtype=gradient.dtype,
        output_shape=shape,
        numpy=numpy,
    )
