"""CuPy implementation of the sign function VJP."""

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


def sign_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    The result is **routed, not multiplied**, for the reason given in the
    NumPy kernel: multiplying the upstream gradient by a materialised
    derivative turns a negative upstream into ``-0.0`` and an infinite or NaN
    upstream into NaN. See docs/sign-semantics.md §6.

    The zero test is performed on the **widened** operand, and that is not a
    stylistic choice. Measured on this toolchain, a binary32 subnormal
    compares equal to zero natively, because flush-to-zero applies to the
    comparison's operand — so testing natively would raise "undefined at
    zero" for a subnormal, which the specification says is an ordinary
    nonzero value. Widening through the PTX conversion first restores the
    distinction.

    That test is the one bounded device synchronisation this VJP performs:
    the comparison and the reduction both run on the device, and only the
    single resulting boolean crosses to the host so the specified
    ``ValueError`` can be raised. No operand value is transferred.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        tested = ieee32.widen(values) if dtype.typecode == "f" else values
        if bool(cupy.any(tested == 0)):
            raise ValueError("sign derivative is undefined at zero")
        # Every result is NaN or +0.0, neither of which is subnormal, so the
        # routing itself needs no format protection.
        result = cupy.where(cupy.isnan(values), values, 0)
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sign VJP kernel returned an unexpected result size")
    return storage
