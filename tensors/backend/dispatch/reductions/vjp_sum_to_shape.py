"""Dispatch for the broadcast-gradient reduction inside the arithmetic VJPs."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

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
    selected = config.get_backend()
    validate_backend_residency((gradient,), selected)
    backend: Any = load_backend(selected)
    result = backend.sum_to_shape(gradient, shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sum_to_shape at dtype "
            f"{gradient.dtype.name} conformingly. This computation runs on the "
            f"selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
