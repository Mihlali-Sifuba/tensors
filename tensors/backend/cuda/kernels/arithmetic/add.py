"""CuPy implementation of addition."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import ieee32
from tensors.backend.cuda.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def add(
    left: Any,
    right: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage at the declared dtype."""
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        # float32 needs the named IEEE instruction to underflow
        # gradually; float64 already does on the device.
        if dtype.typecode == "f":
            result = ieee32.apply("add", left, right)
        else:
            result = cupy.add(left, right)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
