"""Dispatch for reductions, extrema indices, and shape summation."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_sum_products_to_shape(
    gradient: Tensor, factor: Tensor, shape: tuple[int, ...]
) -> Storage:
    """Run the multiply-and-broadcast reduction on the selected backend.

    This is the whole vector-Jacobian product of multiplication, and
    ``Mul.backward`` is its only caller, so it carries the arithmetic
    execution contract directly: the selection decides where it runs, at
    every size, and a backend that declines is reported rather than replaced.
    """
    return run_on_selected_backend(
        "sum_products_to_shape",
        gradient,
        factor,
        shape,
        detail=f"at dtype {gradient.dtype.name}",
    )
