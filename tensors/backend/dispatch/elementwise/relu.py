"""Dispatch for ReLU: the selection alone decides where it runs.

`docs/relu-semantics.md` states the numerical contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. ReLU executes on
the selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element ReLU runs where a
million-element ReLU runs.

Nothing here inspects operand values. No input is a domain error under the
contract — every value, NaN included, has a specified result — so there is no
reduction, no comparison and no device synchronisation on the way through.

``"auto"`` is not a third behaviour. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — so by the time
a call arrives the selection names one backend.
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


def execute_relu(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run ReLU on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
    elif selected == "numpy":
        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. ReLU is unary, so there is no scalar or broadcasting case,
        # and it preserves dtype, so no cast belongs here.
        storage = value._logical_storage_for("numpy")
        lowered = storage.buffer.reshape(value.shape)
    else:
        # The same lowering on the device. Nothing is read back to the host:
        # the reshape stays in device memory, and no operand value is
        # inspected on the way through.
        storage = value._logical_storage_for("cuda")
        lowered = storage.buffer.reshape(value.shape)

    result = backend.relu(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute relu at dtype "
            f"{dtype.name} conformingly. ReLU runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
