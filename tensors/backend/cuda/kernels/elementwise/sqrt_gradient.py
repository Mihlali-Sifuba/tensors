"""CuPy implementation of the square root VJP."""

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


def sqrt_gradient(
    grad_values: Any,
    values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return device storage at the declared dtype.

    The three steps of `docs/sqrt-semantics.md` section 6 are written out in
    order, and for binary32 each one is performed by the governed float32
    arithmetic rather than in a binary64 working precision that is narrowed
    once at the end:

    * the root widens through the PTX conversion, is taken in binary64 and
      narrows back, which section 1.7 established is correctly rounded;
    * the doubling and the division use ``ieee32.apply``, the same
      round-to-nearest PTX instructions the arithmetic contract uses, which
      keep a subnormal upstream gradient and can produce a subnormal result.

    Ordinary CuPy binary32 operations would flush at each of those points.
    ``float64`` needs none of it: the device underflows gradually there and
    its operations are correctly rounded.

    The zero test runs on the **widened** operand for the reason given in
    the sign VJP: a binary32 subnormal compares equal to zero natively, so
    testing natively would raise for an ordinary nonzero primal. That test
    is the one bounded device synchronisation here — one boolean is read
    back, and no operand value is transferred. A negative primal needs no
    test at all: the root is NaN and NaN carries through.
    """
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        binary32 = dtype.typecode == "f"
        tested = ieee32.widen(values) if binary32 else values
        if bool(cupy.any(tested == 0)):
            raise ValueError("sqrt derivative is undefined at zero")
        if binary32:
            root = ieee32.narrow(cupy.sqrt(ieee32.widen(values)))
            two = cupy.full(root.shape, 2.0, dtype=cupy.float32)
            denominator = ieee32.apply("multiply", two, root)
            result = ieee32.apply("divide", grad_values, denominator)
        else:
            root = cupy.sqrt(values)
            denominator = cupy.asarray(2.0, dtype=cupy.float64) * root
            result = grad_values / denominator
    storage = CudaStorage(result, dtype)
    if storage.size != math.prod(output_shape):
        raise RuntimeError("Sqrt VJP kernel returned an unexpected result size")
    return storage
