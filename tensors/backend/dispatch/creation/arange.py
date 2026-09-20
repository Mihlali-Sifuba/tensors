"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def execute_arange(
    start: int | float, step: int | float, count: int, *, dtype: DataType
) -> Storage:
    """Create arithmetic-progression storage on the active backend."""
    return run_on_selected_backend("arange", start, step, count, dtype=dtype)
