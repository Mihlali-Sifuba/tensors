"""Operand marshalling shared by fused forward and backward execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _view

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _fused_arrays(values: Sequence[Tensor], dtype: DataType) -> tuple[Any, ...]:
    """Return contiguous device arrays cast to the fused storage dtype."""
    provider_dtype = cupy.dtype(dtype.name)
    return tuple(
        _view(value).astype(provider_dtype, copy=False).reshape(-1) for value in values
    )
