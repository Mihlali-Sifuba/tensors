"""Dispatch for log: the selection alone decides where it runs.

`docs/log-semantics.md` states the numerical contract this dispatcher must
deliver, and `docs/backends.md`, *Execution requirements*, makes the backend
selection an execution requirement rather than a preference. Log executes on
the selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element log runs where a
million-element log runs.

A non-positive operand is refused, and that is the public contract rather
than an implementation accident: `log` raises `ValueError` where NumPy and
CuPy would return `-inf` or NaN. The refusal is made here, in the one place
that already holds the operand in a native form, rather than in each kernel —
detecting it is backend-specific and written out per branch, but the raise
itself is one statement, so the three selections cannot drift apart in
wording. This follows abs, which refuses its one unrepresentable input the
same way.

**The CUDA check synchronises deliberately.** Deciding whether to raise
needs an answer on the host, so a device reduction runs and one boolean
crosses back. That is the cost of an immediate Python exception and is a
semantic requirement, not an oversight: the alternative is a NaN that
surfaces somewhere else entirely. Only the boolean crosses — no operand
value is materialised — and no other elementwise operation in this package
pays it, because no other one refuses a value.

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


def execute_log(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run log on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    # NaN is not refused: it fails ``<= 0`` and propagates as NaN, which is
    # the IEEE answer and leaves the refusal for values that really are
    # outside the domain. Both zeros are refused, since neither has a
    # logarithm.
    if selected == "python":
        lowered = value._data
        non_positive = any(item <= 0 for item in lowered)
    elif selected == "numpy":
        import numpy

        # Storage owns a flat native buffer and the Tensor owns the layout, so
        # lowering is: take the logical values and give them the Tensor's
        # shape. Log is unary, so there is no scalar or broadcasting case.
        storage = value._logical_storage_for("numpy")
        lowered = storage.buffer.reshape(value.shape)
        non_positive = bool(numpy.any(lowered <= 0))
    else:
        import cupy

        # The same lowering on the device, and no operand value is read back:
        # the comparison and the reduction both run on the device, and only
        # the one resulting boolean crosses to the host so that the specified
        # ValueError can be raised immediately.
        from tensors.backend.cuda.conversion import _widen

        storage = value._logical_storage_for("cuda")
        lowered = storage.buffer.reshape(value.shape)
        non_positive = bool(cupy.any(_widen(lowered) <= 0))

    if non_positive:
        raise ValueError("log is only defined for positive values")

    result = backend.log(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute log at dtype "
            f"{dtype.name} conformingly. Log runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
