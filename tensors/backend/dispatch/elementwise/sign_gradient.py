"""Dispatch for the sign VJP: the selection alone decides where it runs.

`docs/sign-semantics.md` §6 states the contract this dispatcher must deliver,
and `docs/backends.md`, *Execution requirements*, makes the backend selection
an execution requirement rather than a preference. The VJP executes on the
selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies.

Shape and dtype agreement between the upstream gradient and the primal value
is Tensor semantics and is settled by `Sign.backward` before this is reached.
What is settled here is where the work runs and in what representation.
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


def execute_sign_gradient(
    grad: Tensor,
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the sign VJP on the selected backend, or say it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((grad, value), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered_grad = grad._data
        lowered_value = value._data
    elif selected == "numpy":
        # Storage owns a flat native buffer and the Tensor owns the layout,
        # so lowering is: take the logical values and give them the Tensor's
        # shape. The VJP preserves dtype, so no cast belongs here.
        lowered_grad = grad._logical_storage_for("numpy").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("numpy").buffer.reshape(value.shape)
    else:
        # The same lowering on the device; the reshape stays in device memory.
        lowered_grad = grad._logical_storage_for("cuda").buffer.reshape(grad.shape)
        lowered_value = value._logical_storage_for("cuda").buffer.reshape(value.shape)

    result = backend.sign_gradient(
        lowered_grad, lowered_value, dtype=dtype, output_shape=output_shape
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sign_gradient at dtype "
            f"{dtype.name} conformingly. The VJP runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
