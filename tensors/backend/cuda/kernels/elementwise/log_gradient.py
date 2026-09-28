"""CuPy implementation of the natural logarithm VJP."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate, _narrow, _widen
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def log_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    ``G / x``, an ordinary division, governed by
    `docs/arithmetic-semantics.md` section 7.2: the IEEE result stands and
    nothing raises. Dividing once is deliberate — forming ``1 / x`` and
    multiplying rounds twice and loses a digit for no benefit.

    No domain check runs here, so no device synchronisation does either. The
    forward refuses a non-positive operand, and repeating the test would put
    a reduction and a host round trip on every reverse pass.

    Both operands are widened through PTX, because this VJP divides by its
    primal rather than routing it.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        upstream = _widen(grad_values)
        working = _widen(values)
        result = upstream / working
        narrowed = _narrow(result, cupy.dtype(dtype.name))
    storage = CudaStorage(narrowed, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Log VJP kernel returned an unexpected result size")
    return storage
