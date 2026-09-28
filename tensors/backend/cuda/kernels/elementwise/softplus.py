"""CuPy implementation of softplus."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def softplus(
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    ``log1p(exp(-|x|)) + max(x, 0)``, the form the other two backends use and
    for the same range reason. What is specific here is the two format
    crossings: ``astype`` flushes a binary32 subnormal in both directions on
    this toolchain, and softplus reaches that band for a large negative
    input, where its result is ``exp(x)`` itself.

    ``_widen`` and ``_narrow`` are no-ops for any dtype that is not binary32,
    so an integer operand promoted to ``float64`` by the operation passes
    through them unchanged.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        working = _widen(cupy.asarray(values))
        result = cupy.log1p(cupy.exp(-cupy.abs(working))) + cupy.maximum(working, 0.0)
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Softplus kernel returned an unexpected result size")
    return storage
