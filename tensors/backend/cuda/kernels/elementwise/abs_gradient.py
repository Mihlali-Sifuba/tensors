"""CuPy implementation of the absolute-value VJP."""

from __future__ import annotations
import math
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import ieee32
from tensors.backend.cuda.storage import CudaStorage
from tensors.backend.storage import Storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def abs_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    **Routing, not multiplication**, for the reason given in the NumPy
    kernel. See docs/abs-semantics.md section 6.

    binary32 is routed in binary64, and the measurement that forces it is
    narrower than it first appears. A ``where`` selection preserves a
    subnormal, because selecting is not arithmetic — but **negating** one
    does not: ``-g`` on a native binary32 subnormal was measured returning
    minus or plus zero. The negative branch is exactly where this VJP
    negates, so an upstream subnormal would be lost there and nowhere else.
    Widening through the PTX conversion, routing, and narrowing back keeps
    it.

    ``float64`` needs none of that: the device underflows gradually there.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            upstream = ieee32.widen(grad_values)
            primal = ieee32.widen(values)
        else:
            upstream = grad_values
            primal = values
        rectified = cupy.where(
            primal > 0,
            upstream,
            cupy.where(primal < 0, -upstream, 0),
        )
        result = cupy.where(cupy.isnan(primal), primal, rectified)
        if dtype.typecode == "f":
            result = ieee32.narrow(result)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Abs VJP kernel returned an unexpected result size")
    return storage
