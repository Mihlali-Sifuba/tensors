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
from typing import Any
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.preparation import BinaryExecution
from tensors.backend.validation import validate_backend_residency


def execute_divide(request: BinaryExecution) -> Storage:
    """Run divide on the selected backend, or report that it cannot run there."""
    backend: Any = load_backend(request.backend)
    result = backend.divide(
        request.left,
        request.right,
        dtype=request.dtype,
        output_shape=request.output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {request.backend} backend cannot execute divide at dtype "
            f"{request.dtype.name} conformingly. Arithmetic runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), request.backend)
    return result
