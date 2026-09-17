"""Arithmetic dispatch: where ``+``, ``-``, ``*`` and ``/`` execute.

The policy is deterministic: arithmetic executes on the selected backend, and
on no other. `docs/backends.md`, *Execution requirements*, makes that a
requirement rather than a preference, so a backend that cannot produce the
required result raises instead of handing the work away quietly.

``"auto"`` is not a third behaviour. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — and from here
on it is indistinguishable from having named that backend. There is no
workload-size threshold and no small-tensor case: where an operation runs is
decided by the selection alone, never by the shape of its operands.

Only the four contract operations dispatch through here. Forward power no
longer consults the workload policy either, but keeps its own dispatcher: its
kernels use a decline to signal a domain error as well as an unsupported
operand, so it cannot raise on one without losing the other. See
``dispatch/arithmetic/power.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from tensors.backend.config import BackendOperationUnsupportedError, get_backend
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _reference(name: str) -> Callable[..., Storage]:
    """Return the Python reference kernel for one arithmetic operation."""
    module = __import__(
        f"tensors.backend.python.kernels.arithmetic.{name}",
        fromlist=[name],
    )
    return getattr(module, name)


def execute_arithmetic(
    name: str,
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Execute one arithmetic operation on the selected backend."""
    selected = get_backend()
    if selected == "python":
        return _reference(name)(left, right, dtype=dtype, output_shape=output_shape)

    backend: Any = load_backend(selected)
    result = getattr(backend, name)(left, right, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute {name} at dtype "
            f"{dtype.name} conformingly. Arithmetic runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    return result


__all__ = ["execute_arithmetic"]
