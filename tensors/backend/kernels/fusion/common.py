"""Operand marshalling shared by fused forward and backward execution."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TYPE_CHECKING, cast

from ..core import _view

if TYPE_CHECKING:
    from ....dtype import DataType
    from ....tensor import Tensor

def _fused_arrays(
    values: Sequence[Tensor],
    dtype: DataType,
    cupy: Any,
) -> tuple[Any, ...]:
    """Return contiguous device arrays cast to the fused storage dtype."""
    provider_dtype = cupy.dtype(dtype.name)
    return tuple(
        _view(value, cupy).astype(provider_dtype, copy=False).reshape(-1)
        for value in values
    )
