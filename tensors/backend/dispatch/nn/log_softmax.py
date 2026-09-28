"""Strict selected-backend dispatch for log-softmax."""

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


def execute_log_softmax(value: Tensor, axis: int, *, dtype: DataType) -> Storage:
    """Run log-softmax on the selected backend without size-based fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)
    buffer = value._logical_storage_for(selected).buffer
    values = buffer if selected == "python" else buffer.reshape(value.shape)
    result = backend.log_softmax(values, value.shape, axis, dtype=dtype)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute log_softmax at dtype "
            f"{dtype.name} conformingly. Log-softmax runs on the selected "
            "backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
