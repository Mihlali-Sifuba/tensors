"""Strict dispatch for greater."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_greater(
    left: Tensor,
    right: Tensor,
    *,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run greater on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((left, right), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        from tensors.utils.broadcasting import broadcast_to

        lowered_left = broadcast_to(left, output_shape)._data
        lowered_right = broadcast_to(right, output_shape)._data
    elif selected == "numpy":
        lowered_left = left._logical_storage_for("numpy").buffer.reshape(left.shape)
        lowered_right = right._logical_storage_for("numpy").buffer.reshape(right.shape)
    else:
        lowered_left = left._logical_storage_for("cuda").buffer.reshape(left.shape)
        lowered_right = right._logical_storage_for("cuda").buffer.reshape(right.shape)

    result = backend.greater(
        lowered_left,
        lowered_right,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute greater conformingly. "
            f"greater runs on the selected backend; select another backend "
            f"to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
