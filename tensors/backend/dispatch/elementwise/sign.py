"""Dispatch for sign: the selection alone decides where it runs.

`docs/sign-semantics.md` states the numerical contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. Sign executes on
the selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element sign runs where a
million-element sign runs.

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


def execute_sign(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run sign on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
    elif selected == "numpy":
        import numpy

        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. Sign is unary, so there is no scalar or broadcasting case.
        # Sign also preserves dtype, so the logical buffer already carries the
        # declared one; the cast below is a guard, not a conversion.
        native = numpy.dtype(dtype.name)
        storage = value._logical_storage_for("numpy")
        lowered = storage.buffer.reshape(value.shape)
        if lowered.dtype != native:
            lowered = lowered.astype(native, copy=False)
    else:
        import cupy

        # The same lowering on the device. Nothing is read back to the host:
        # reshape and astype both stay in device memory.
        native = cupy.dtype(dtype.name)
        storage = value._logical_storage_for("cuda")
        lowered = storage.buffer.reshape(value.shape)
        if lowered.dtype != native:
            lowered = lowered.astype(native, copy=False)

    result = backend.sign(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sign at dtype "
            f"{dtype.name} conformingly. Sign runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
