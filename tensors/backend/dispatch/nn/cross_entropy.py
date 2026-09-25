"""Strict selected-backend dispatch for cross-entropy."""

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


def execute_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    axis: int,
    *,
    reduction: LossReduction,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run dense cross-entropy on the selected backend without fallback."""
    selected = config.get_backend()
    validate_backend_residency((logits, targets), selected)
    backend: Any = load_backend(selected)
    logits_buffer = logits._logical_storage_for(selected).buffer
    targets_buffer = targets._logical_storage_for(selected).buffer
    logits_values = (
        logits_buffer if selected == "python" else logits_buffer.reshape(logits.shape)
    )
    target_values = (
        targets_buffer
        if selected == "python"
        else targets_buffer.reshape(targets.shape)
    )
    result = backend.cross_entropy(
        logits_values,
        target_values,
        logits.shape,
        targets.shape,
        axis,
        reduction=reduction,
        dtype=dtype,
        logits_dtype=logits.dtype,
        output_shape=output_shape,
    )
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute cross_entropy at dtype "
            f"{dtype.name} conformingly. The loss runs on the selected backend; "
            "select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
