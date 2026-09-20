"""Dispatch for multiply: the selection alone decides where it runs.

`docs/backends.md`, *Execution requirements*, makes the backend selection an
execution requirement rather than a preference. Multiplication executes on the
selected backend's kernel, never on another's, and a backend that cannot
produce the required result raises instead of handing the work away quietly.
No workload-size policy applies: a one-element multiply runs where a
million-element multiply runs.

``"auto"`` is not a third behaviour. It resolves to a concrete backend when it
is selected — NumPy when NumPy is installed, Python otherwise — so by the time
a call arrives the selection names one backend.
"""

from __future__ import annotations

from itertools import repeat
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.backend.storage import Storage
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_multiply(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run multiply on the selected backend, or report that it cannot run there."""
    selected = config.get_backend()
    validate_backend_residency((left, right), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        from tensors.tensor import Tensor
        from tensors.utils.broadcasting import broadcast_to

        left_is_tensor = isinstance(left, Tensor)
        right_is_tensor = isinstance(right, Tensor)
        if left_is_tensor and right_is_tensor:
            lowered_left = broadcast_to(left, output_shape)._data
            lowered_right = broadcast_to(right, output_shape)._data
        elif left_is_tensor:
            lowered_left = left._data
            lowered_right = repeat(right)
        elif right_is_tensor:
            lowered_left = repeat(left)
            lowered_right = right._data
        else:
            lowered_left = (left,)
            lowered_right = (right,)
    elif selected == "numpy":
        from tensors.backend.numpy.conversion import _arithmetic_operand

        lowered_left = _arithmetic_operand(left, dtype)
        lowered_right = _arithmetic_operand(right, dtype)
    else:
        from tensors.backend.cuda.conversion import _arithmetic_operand

        lowered_left = _arithmetic_operand(left, dtype)
        lowered_right = _arithmetic_operand(right, dtype)

    result = backend.multiply(
        lowered_left,
        lowered_right,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute multiply at dtype "
            f"{dtype.name} conformingly. Arithmetic runs on the selected "
            f"backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
