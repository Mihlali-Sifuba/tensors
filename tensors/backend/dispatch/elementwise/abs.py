"""Dispatch for abs: the selection alone decides where it runs.

`docs/abs-semantics.md` states the numerical contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. Abs executes on
the selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element abs runs where a
million-element abs runs.

One input is specified to fail. The magnitude of a signed integer dtype's
least value is one past its greatest, so `abs(-128)` has no `int8` result.
That is refused here, in the one place that already holds the operand in a
native form, rather than left to a typed buffer to discover. Detecting it is
backend-specific and written out per branch; the refusal itself is one raise,
so the three selections cannot drift apart in wording.

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
from tensors.dtype import _INTEGER_LIMITS

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_abs(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run abs on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    # Only a signed integer dtype has an unrepresentable magnitude. uint8's
    # lower bound is zero, and a floating dtype is symmetric.
    minimum: int | None = None
    if dtype.kind == "integer":
        lower, _ = _INTEGER_LIMITS[dtype.typecode]
        if lower < 0:
            minimum = lower

    if selected == "python":
        lowered = value._data
        unrepresentable = minimum is not None and any(
            item == minimum for item in lowered
        )
    elif selected == "numpy":
        import numpy

        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. Abs is unary, so there is no scalar or broadcasting case.
        storage = value._logical_storage_for("numpy")
        lowered = storage.buffer.reshape(value.shape)
        unrepresentable = minimum is not None and bool(numpy.any(lowered == minimum))
    else:
        import cupy

        # The same lowering on the device, and no operand value is read back:
        # the comparison and the reduction both run on the device, and only
        # the one resulting boolean crosses to the host so that the specified
        # OverflowError can be raised immediately.
        storage = value._logical_storage_for("cuda")
        lowered = storage.buffer.reshape(value.shape)
        unrepresentable = minimum is not None and bool(cupy.any(lowered == minimum))

    if unrepresentable:
        raise OverflowError(f"abs({minimum}) is not representable in {dtype.name}")

    result = backend.abs(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute abs at dtype "
            f"{dtype.name} conformingly. Abs runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
