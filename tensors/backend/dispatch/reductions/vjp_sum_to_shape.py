"""Dispatch for the broadcast-gradient reduction inside the arithmetic VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_vjp_sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage:
    """Reduce a broadcast gradient on the selected backend.

    The same computation as ``execute_sum_to_shape``, under the execution
    contract of `docs/backends.md`: no workload-size policy decides where it
    runs, and a backend that declines is reported rather than replaced by the
    Python reference.

    The addition and subtraction VJPs use this one, because ``+`` and ``-``
    are inside the arithmetic contract. ``execute_sum_to_shape`` still serves
    the divide, power, where, loss and extremum backward passes, which are
    not, and which keep the policy and the fallback until that contract
    reaches them. That is the only reason both exist.
    """
    from tensors.backend.python.kernels.reductions.sum_to_shape import (
        sum_to_shape as reference,
    )

    return run_on_selected_backend(
        "sum_to_shape",
        reference,
        gradient,
        shape,
        detail=f"at dtype {gradient.dtype.name}",
    )
