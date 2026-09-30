"""Deterministic floating contraction for the NumPy backend.

A contraction is a product then a sum. Each product is formed once, rounded to
the operands' dtype, and the products of one output element are reduced in
increasing contraction-index order with the package's pairwise tree
(`docs/summation-semantics.md`). Provider matrix products are not used: their
blocking and fused multiply-adds would choose a different summation.
"""

from __future__ import annotations

from typing import Any

import numpy

from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.kernels.reductions.pairwise import pairwise_float_sum


def pairwise_matmul(left: Any, right: Any) -> Any:
    """Contract ``[..., m, k] @ [..., k, n]`` as products then a pairwise sum.

    Each product is formed once in the operands' dtype, and the ``k``
    products of an output element are reduced in increasing ``k`` order with
    :func:`pairwise_float_sum`. No fused multiply-add is involved: NumPy
    multiplies and adds in separate rounded operations.
    """
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        products = left[..., :, :, None] * right[..., None, :, :]
    return numpy.squeeze(pairwise_float_sum(products, (products.ndim - 2,)), axis=-2)


__all__ = ["pairwise_matmul"]
