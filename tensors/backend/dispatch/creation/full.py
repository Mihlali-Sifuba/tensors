"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def execute_full(
    shape: tuple[int, ...], fill_value: int | float, *, dtype: DataType
) -> Storage:
    """Create constant-filled storage on the active backend."""
    from tensors.backend.python.kernels.creation.full import full as reference

    return run_on_selected_backend(
        "full", reference, shape, fill_value, dtype=dtype
    )
