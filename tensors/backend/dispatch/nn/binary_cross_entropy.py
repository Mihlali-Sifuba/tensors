"""Strict selected-backend dispatch for binary cross-entropy."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.backend.types import LossReduction
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_binary_cross_entropy(
    prediction: Tensor,
    target: Tensor,
    *,
    from_logits: bool,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run binary cross-entropy on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((prediction, target), selected)
    backend: Any = load_backend(selected)
    prediction_buffer = prediction._logical_storage_for(selected).buffer
    target_buffer = target._logical_storage_for(selected).buffer
    prediction_values = (
        prediction_buffer
        if selected == "python"
        else prediction_buffer.reshape(prediction.shape)
    )
    target_values = (
        target_buffer if selected == "python" else target_buffer.reshape(target.shape)
    )
    result = backend.binary_cross_entropy(
        prediction_values,
        target_values,
        prediction.shape,
        target.shape,
        from_logits=from_logits,
        reduction=reduction,
        dtype=dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute binary_cross_entropy at "
            f"dtype {dtype.name} conformingly. The loss runs on the selected "
            "backend; select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
