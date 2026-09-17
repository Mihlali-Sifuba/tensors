"""Dedicated arithmetic dispatch for divide.

Explicit selection decides where this runs; see :mod:`_execution`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
from tensors.backend.dispatch.arithmetic._execution import execute_arithmetic
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def execute_divide(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Run divide on the selected backend."""
    return execute_arithmetic(
        "divide", left, right, dtype=dtype, output_shape=output_shape
    )
