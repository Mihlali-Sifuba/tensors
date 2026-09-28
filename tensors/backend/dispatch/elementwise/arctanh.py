"""Strict dispatch for arctanh."""

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


def execute_arctanh(
    value: Tensor,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run arctanh on the selected backend without cross-backend fallback."""
    selected = config.get_backend()
    validate_backend_residency((value,), selected)
    backend: Any = load_backend(selected)

    if selected == "python":
        import math

        lowered = value._data
        outside = any(
            not math.isnan(float(item)) and not -1.0 < item < 1.0 for item in lowered
        )
    elif selected == "numpy":
        import numpy

        lowered = value._logical_storage_for("numpy").buffer.reshape(value.shape)
        outside = bool(
            numpy.any(~numpy.isnan(lowered) & ((lowered <= -1.0) | (lowered >= 1.0)))
        )
    else:
        import cupy

        lowered = value._logical_storage_for("cuda").buffer.reshape(value.shape)
        outside = bool(
            cupy.any(~cupy.isnan(lowered) & ((lowered <= -1.0) | (lowered >= 1.0)))
        )

    if outside:
        raise ValueError("arctanh is only defined for values strictly between -1 and 1")

    result = backend.arctanh(lowered, dtype=dtype, output_shape=output_shape)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute arctanh at dtype "
            f"{dtype.name} conformingly. arctanh runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
