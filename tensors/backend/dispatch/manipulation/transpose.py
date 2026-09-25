"""Strict selected-backend dispatch for axis permutation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_transpose(
    value: Tensor, permutation: tuple[int, ...], *, output_shape: tuple[int, ...]
) -> Storage:
    """Copy an axis permutation on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    buffer = value._logical_storage_for(selected).buffer
    lowered = buffer if selected == "python" else buffer.reshape(value.shape)
    result = backend.transpose(
        lowered,
        permutation,
        input_shape=tuple(value.shape),
        dtype=value.dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute transpose at dtype "
            f"{value.dtype.name} conformingly. This computation runs on the "
            "selected backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
