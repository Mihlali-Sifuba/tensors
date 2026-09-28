"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
from tensors.backend import config
from tensors.backend.config import BackendOperationUnsupportedError
from tensors.backend.loading import load_backend
from tensors.backend.storage import Storage
from tensors.backend.validation import validate_backend_residency

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage:
    """Expand class-index targets on the selected backend."""
    selected = config.get_backend()
    validate_backend_residency((logits, targets), selected)
    backend: Any = load_backend(selected)
    result = backend.one_hot_targets(logits, targets, axis)
    if result is None:
        raise BackendOperationUnsupportedError(
            f"The {selected} backend cannot execute one_hot_targets "
            f"conformingly. This computation runs on the selected backend; "
            f"select another backend to run it elsewhere."
        )
    validate_backend_residency((result,), selected)
    return result
