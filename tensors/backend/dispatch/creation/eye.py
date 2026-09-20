"""Dispatch for values built from parameters rather than transformed."""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch._selected import run_on_selected_backend
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def execute_eye(rows: int, columns: int, k: int, *, dtype: DataType) -> Storage:
    """Create identity-like matrix storage on the active backend."""
    return run_on_selected_backend("eye", rows, columns, k, dtype=dtype)
