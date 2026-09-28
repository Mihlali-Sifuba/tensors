"""Strict dispatch for tan."""

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


def execute_tan(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run tan on the selected backend without cross-backend fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        lowered = value._data
        import math

        invalid = any(math.isinf(float(item)) for item in lowered)
    elif selected == "numpy":
        import numpy

        lowered = value._logical_storage_for("numpy").buffer.reshape(value.shape)
        invalid = bool(numpy.any(numpy.isinf(lowered)))
    else:
        import cupy

        lowered = value._logical_storage_for("cuda").buffer.reshape(value.shape)
        invalid = bool(cupy.any(cupy.isinf(lowered)))

    if invalid:
        raise ValueError("math domain error")

    result = backend.tan(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute tan at dtype "
            f"{dtype.name} conformingly. tan runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
