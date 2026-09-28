"""CuPy implementation of the ReLU VJP."""

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


def relu_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    **Routing, not multiplication**, so an infinite or NaN upstream on the
    inactive side cannot manufacture a NaN. See docs/relu-semantics.md
    section 6.

    Only the **predicate** is widened, and that is the whole of what binary32
    needs here. Measured on this toolchain, ``x > 0`` is false for every
    positive binary32 subnormal, because flush-to-zero reaches the
    comparison's operand — so the entire subnormal band would be routed to
    the inactive branch. Widening through the PTX conversion restores it.

    The upstream gradient is *not* widened, deliberately: this VJP only ever
    selects it, and a selection is not arithmetic, so a subnormal upstream
    passes through a native ``where`` unchanged. That is what separates this
    kernel from the abs VJP, which negates on one branch and therefore has
    to route in binary64.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        primal = ieee32.widen(values) if dtype.typecode == "f" else values
        rectified = cupy.where(primal > 0, grad_values, 0)
        result = cupy.where(cupy.isnan(primal), values, rectified)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("ReLU VJP kernel returned an unexpected result size")
    return storage
