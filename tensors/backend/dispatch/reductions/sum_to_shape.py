"""Strict selected-backend dispatch for broadcast-gradient summation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def execute_sum_to_shape(gradient: Tensor, shape: tuple[int, ...]) -> Storage:
    """Reduce a broadcast gradient on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((gradient,), selected)
    backend: Any = load_backend(selected)
    buffer = gradient._logical_storage_for(selected).buffer
    values = buffer if selected == "python" else buffer.reshape(gradient.shape)
    result = backend.sum_to_shape(values, gradient.shape, shape, dtype=gradient.dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute sum_to_shape conformingly. "
            "The reduction runs on the selected backend; select another backend "
            "to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
