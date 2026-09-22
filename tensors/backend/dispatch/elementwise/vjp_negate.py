"""Dispatch for the negation inside the subtraction VJP."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

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
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    result = backend.negate(value, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute negate at dtype "
            f"{dtype.name} conformingly. This computation runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
