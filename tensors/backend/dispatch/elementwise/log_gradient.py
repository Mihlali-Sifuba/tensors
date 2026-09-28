"""Dispatch for the log VJP: the selection alone decides where it runs.

`docs/log-semantics.md` section 6 states the contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. No workload-size
policy applies and there is no Python-reference fallback.

**No domain check runs here, unlike the forward.** The VJP is ``G / x``, an
ordinary division, and
[arithmetic semantics 7.2](../../../../docs/arithmetic-semantics.md#72-division-by-zero)
governs it: floating division adopts the IEEE result and does not raise. The
domain is enforced once, by the forward, which cannot have produced a primal
outside it; a reverse pass therefore pays no reduction and no device
synchronisation, where repeating the check would cost one on every backward.

A primal that never went through the forward — reached by calling the VJP
directly — divides under IEEE like any other division rather than being
refused a second time.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_log_gradient(
    grad: Tensor,
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the log VJP on the selected backend, or say it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((grad, value), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered_grad = grad._data
        lowered_value = value._data
    elif selected == "numpy":
        lowered_grad = grad._logical_storage_for("numpy").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("numpy").buffer.reshape(value.shape)
    else:
        lowered_grad = grad._logical_storage_for("cuda").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("cuda").buffer.reshape(value.shape)

    result = backend.log_gradient(
        lowered_grad, lowered_value, dtype=dtype, output_shape=output_shape
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute log_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
