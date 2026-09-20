"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def execute_one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Storage:
    """Expand class-index targets on the active backend."""
    from tensors.backend.python.kernels.creation.one_hot_targets import (
        one_hot_targets as reference,
    )

    return run_on_selected_backend(
        "one_hot_targets", reference, logits, targets, axis
    )
