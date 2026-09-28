"""Strict selected-backend dispatch for concatenation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_concat(
    values: Sequence[Tensor],
    axis: int,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Concatenate on the selected backend without size-based fallback."""
    selected = config.get_backend()
    validate_backend_residency(values, selected)
    backend: Any = load_backend(selected)
    lowered = []
    shapes = []
    for value in values:
        buffer = value._logical_storage_for(selected).buffer
        lowered.append(buffer if selected == "python" else buffer.reshape(value.shape))
        shapes.append(tuple(value.shape))
    result = backend.concat(
        tuple(lowered),
        tuple(shapes),
        axis,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute concat at dtype "
            f"{dtype.name} conformingly. This computation runs on the selected "
            "backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
