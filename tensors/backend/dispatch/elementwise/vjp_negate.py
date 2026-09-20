"""Dispatch for the negation inside the subtraction VJP."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_vjp_negate(value: Tensor, *, dtype: DataType) -> Storage:
    """Negate on the selected backend, for the subtraction VJP.

    The gradient of ``a - b`` with respect to ``b`` is the negated upstream
    gradient, which puts this negation inside the arithmetic contract: it
    executes where the selection says, at any size, or reports that it
    cannot. Forward negation is a separate operation and keeps
    ``execute_negate``, workload policy and reference fallback included.
    """
    return run_on_selected_backend(
        "negate",
        value,
        dtype=dtype,
        detail=f"at dtype {dtype.name}",
    )
