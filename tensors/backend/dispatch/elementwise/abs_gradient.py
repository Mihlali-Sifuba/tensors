"""Dispatch for the absolute-value VJP: the selection decides where it runs.

`docs/abs-semantics.md` §6 states the contract this dispatcher must deliver,
and `docs/backends.md`, *Execution requirements*, makes the backend selection
an execution requirement rather than a preference. No workload-size policy
applies and there is no Python-reference fallback.

Nothing here inspects operand values. The absolute value's VJP has no domain
error — the kink uses a chosen subgradient rather than raising — so unlike
the sign VJP this dispatcher needs no reduction and no device
synchronisation.
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


def execute_abs_gradient(
    grad: Tensor,
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the abs VJP on the selected backend, or say it cannot run there."""
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

    result = backend.abs_gradient(
        lowered_grad, lowered_value, dtype=dtype, output_shape=output_shape
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute abs_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
