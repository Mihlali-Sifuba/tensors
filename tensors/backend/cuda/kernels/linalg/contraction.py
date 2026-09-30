"""Deterministic floating contraction for the CUDA backend.

A contraction is a product then a sum. Each product is formed once, rounded to
the operands' dtype, and the products of one output element are reduced in
increasing contraction-index order with the package's pairwise tree
(`docs/summation-semantics.md`). Provider matrix products are not used: their
blocking and fused multiply-adds would choose a different summation.
"""

from __future__ import annotations

from typing import Any

import cupy

from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.reductions.pairwise import _multiply
from tensors.backend.cuda.kernels.reductions.pairwise import pairwise_float_sum


def pairwise_matmul(left: Any, right: Any) -> Any:
    """Contract ``[..., m, k] @ [..., k, n]`` as products then a pairwise sum.

    Each product is formed once in the operands' dtype and the ``k``
    products of an output element are reduced in increasing ``k`` order with
    :func:`pairwise_float_sum`.
    """
    left_factors, right_factors = cupy.broadcast_arrays(
        left[..., :, :, None], right[..., None, :, :]
    )
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        products = _multiply(left_factors, right_factors)
    return cupy.squeeze(pairwise_float_sum(products, (products.ndim - 2,)), axis=-2)


__all__ = ["pairwise_matmul"]
