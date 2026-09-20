"""Dispatch for divide: the selection alone decides where it runs.

`docs/backends.md`, *Execution requirements*, makes the backend selection an
execution requirement rather than a preference. Division executes on the
selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element divide runs where a
million-element divide runs.

``"auto"`` is not a third behaviour. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — so by the time
a call arrives the selection names one backend.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_operands, validate_result

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_divide(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run divide on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_operands((left, right), selected, context="divide")
    if selected == "python":
        from tensors.backend.python.kernels.arithmetic.divide import (
            divide as reference,
        )

        return validate_result(
            reference(left, right, dtype=dtype, output_shape=output_shape),
            selected,
            context="divide",
        )

    backend: Any = load_backend(selected)
    result = backend.divide(left, right, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute divide at dtype "
            f"{dtype.name} conformingly. Arithmetic runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    return validate_result(result, selected, context="divide")
