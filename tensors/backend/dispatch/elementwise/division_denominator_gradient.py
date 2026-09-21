"""Dispatch for the division-denominator VJP.

`docs/arithmetic-semantics.md` section 7.4 states the contract this
dispatcher must deliver, and `docs/backends.md`, *Execution requirements*,
makes the backend selection an execution requirement rather than a
preference. No workload-size policy applies and there is no
Python-reference fallback.

Nothing here inspects operand values. A zero denominator is not an error in
this VJP — section 7.2 gives it a signed infinity or a NaN — so there is no
reduction and no device synchronisation, and the kernels answer it
numerically.

The three operands arrive already broadcast to one shape, which is Tensor
semantics settled by the operation layer. What is settled here is where the
work runs and in what representation.
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


def execute_division_denominator_gradient(
    grad: Tensor,
    numerator: Tensor,
    denominator: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run the denominator VJP on the selected backend, or report it cannot."""
    selected = config.get_backend()
    validate_backend_residency((grad, numerator, denominator), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered_grad = grad._data
        lowered_numerator = numerator._data
        lowered_denominator = denominator._data
    elif selected == "numpy":
        import numpy

        # Storage owns a flat native buffer and the Tensor owns the layout,
        # so lowering is: take the logical values, give them the Tensor's
        # shape, and place them in the declared dtype.
        native = numpy.dtype(dtype.name)

        def lower(operand: Tensor) -> Any:
            buffer = operand._logical_storage_for("numpy").buffer
            lowered = buffer.reshape(operand.shape)
            if lowered.dtype != native:
                lowered = lowered.astype(native, copy=False)
            return lowered

        lowered_grad = lower(grad)
        lowered_numerator = lower(numerator)
        lowered_denominator = lower(denominator)
    else:
        import cupy

        # The same lowering on the device; nothing is read back to the host.
        native = cupy.dtype(dtype.name)

        def lower(operand: Tensor) -> Any:
            buffer = operand._logical_storage_for("cuda").buffer
            lowered = buffer.reshape(operand.shape)
            if lowered.dtype != native:
                lowered = lowered.astype(native, copy=False)
            return lowered

        lowered_grad = lower(grad)
        lowered_numerator = lower(numerator)
        lowered_denominator = lower(denominator)

    result = backend.division_denominator_gradient(
        lowered_grad,
        lowered_numerator,
        lowered_denominator,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute "
            f"division_denominator_gradient at dtype {dtype.name} "
            f"conformingly. The VJP runs on the selected backend; select "
            f"another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
