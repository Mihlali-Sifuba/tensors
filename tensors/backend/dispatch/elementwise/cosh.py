"""Strict dispatch for cosh."""

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


def execute_cosh(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run cosh on the selected backend without cross-backend fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
    elif selected == "numpy":
        lowered = value._logical_storage_for("numpy").buffer.reshape(value.shape)
    else:
        lowered = value._logical_storage_for("cuda").buffer.reshape(value.shape)

    result = backend.cosh(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute cosh at dtype "
            f"{dtype.name} conformingly. cosh runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
