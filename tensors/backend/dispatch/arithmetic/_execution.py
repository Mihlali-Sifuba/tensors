"""Arithmetic dispatch: where ``+``, ``-``, ``*`` and ``/`` execute.

`docs/backends.md`, *Execution requirements*, makes explicit selection an
execution requirement rather than a performance preference. Under an explicit
selection a supported operation executes on that backend's kernel, never on
another's, and a backend that cannot execute it raises instead of handing the
work away quietly. Workload-size policy may decide *how* an operation runs, but
never *where*, so the size thresholds do not apply here.

Automatic selection keeps both freedoms. It may consult the workload policy and
may run the Python reference instead, because after the arithmetic refactor
that path satisfies the same numerical contract: integers wrap at their
declared width and binary32 results are correctly rounded, the double rounding
through binary64 being harmless for these four operations
(`docs/arithmetic-semantics.md` section 5.3).

Only the four contract operations dispatch through here. Power keeps the older
arrangement, where declining to the reference is still how a kernel reports
that it cannot produce the required result.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from tensors.backend.config import (
    BackendOperationUnsupportedError,
    get_backend,
    selection_is_automatic,
)
from tensors.backend.loading import load_backend
from tensors.backend.policy import _shape_size, should_accelerate_elementwise
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
    """Execute one arithmetic operation where the selection requires."""
    selected = get_backend()
    if selected == "python":
        return _reference(name)(left, right, dtype=dtype, output_shape=output_shape)

    if selection_is_automatic() and not should_accelerate_elementwise(
        selected, _shape_size(output_shape)
    ):
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
