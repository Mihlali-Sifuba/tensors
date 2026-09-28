"""Dispatch for softplus: the selection alone decides where it runs.

`docs/backends.md`, *Execution requirements*, makes the backend selection an
execution requirement rather than a preference. Softplus executes on the
selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element softplus runs where a
million-element softplus runs.

Under the policy this replaced, a NumPy tensor of fewer than 32 elements ran
the Python reference and came back in ``PythonStorage``, so an activation in
a small network changed the residency of everything downstream of it.

Nothing here inspects operand values. Softplus has no domain error — every
input, NaN and both infinities included, has a specified result — so there is
no reduction, no comparison and no device synchronisation on the way through.

The operand's dtype and the result's are separate. An integer operand is
promoted to ``float64`` by :class:`~tensors.operations.activations.softplus.Softplus`,
which is where that decision belongs; this only carries the answer down.
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


def execute_softplus(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run softplus on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
    elif selected == "numpy":
        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. Softplus is unary, so there is no scalar or broadcasting case.
        lowered = value._logical_storage_for("numpy").buffer.reshape(value.shape)
    else:
        # The same lowering on the device. Nothing is read back to the host:
        # the reshape stays in device memory, and no operand value is
        # inspected on the way through.
        lowered = value._logical_storage_for("cuda").buffer.reshape(value.shape)

    result = backend.softplus(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute softplus at dtype "
            f"{dtype.name} conformingly. Softplus runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
