"""Sound NumPy-native floating contraction primitives."""

from __future__ import annotations

from typing import Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.reductions.exact import certified_float_sum


def certified_matmul(left: Any, right: Any) -> Any | None:
    """Return a conforming matrix product, or decline without guessing."""
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        left_factors = left[..., :, :, None]
        right_factors = right[..., None, :, :]
        products = left_factors * right_factors
        finite_factors = numpy.isfinite(left_factors) & numpy.isfinite(right_factors)
        range_lost = finite_factors & (
            numpy.isinf(products)
            | ((products == 0.0) & (left_factors != 0.0) & (right_factors != 0.0))
        )
    if bool(numpy.any(range_lost)):
        return None
    if products.shape[-2] == 0:
        return numpy.sum(products, axis=-2)
    result = certified_float_sum(products, (-2,))
    return None if result is None else numpy.squeeze(result, axis=-2)


__all__ = ["certified_matmul"]
