"""Dispatch for sqrt: the selection alone decides where it runs.

`docs/sqrt-semantics.md` states the numerical contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. Sqrt executes on
the selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element sqrt runs where a
million-element sqrt runs.

Nothing here inspects operand values. A negative operand is not an error
under the contract — it has a specified NaN result — so there is no domain
check, and therefore no reduction and no device synchronisation. That is the
opposite of abs, whose one unrepresentable input has to be found before a
kernel runs.

The lowering hands each kernel the operand in the Tensor's *own* dtype. Sqrt
is the first migrated operation whose output dtype can differ from its input
dtype — an integer operand yields `float64` — and the conversion is made
explicitly in each kernel rather than hidden in a cast here.

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


def execute_sqrt(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run sqrt on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
    elif selected == "numpy":
        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. Sqrt is unary, so there is no scalar or broadcasting case.
        storage = value._logical_storage_for("numpy")
        lowered = storage.buffer.reshape(value.shape)
    else:
        # The same lowering on the device. Nothing is read back to the host:
        # the reshape stays in device memory, and no operand value is
        # inspected on the way through.
        storage = value._logical_storage_for("cuda")
        lowered = storage.buffer.reshape(value.shape)

    result = backend.sqrt(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sqrt at dtype "
            f"{dtype.name} conformingly. Sqrt runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
