"""CuPy implementation of exponentiation."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.kernels.arithmetic import ieee32
from tensors.backend.cuda.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


#: The unsigned dtype of each width. Modular arithmetic is defined for
#: unsigned integers, and the bit pattern of a two's-complement signed value
#: is the same, so the whole computation runs unsigned and is reinterpreted at
#: the end rather than relying on signed overflow, which C leaves undefined.
_UNSIGNED_OF_WIDTH = {1: "uint8", 2: "uint16", 4: "uint32", 8: "uint64"}


def _integer_power(left, right, native):
    """Exponentiation by squaring in a fixed width (section 12.4.1).

    The exponent is non-negative: section 12.4.2 rejects a negative one before
    evaluation. Each iteration handles one bit of the exponent, so the work is
    O(log n) elementwise multiplications rather than n of them, and every
    product wraps at the declared width because it is computed unsigned.
    """
    unsigned = cupy.dtype(_UNSIGNED_OF_WIDTH[native.itemsize])

    base = cupy.ascontiguousarray(cupy.asarray(left, dtype=native))
    exponent = cupy.ascontiguousarray(cupy.asarray(right, dtype=native))
    base, exponent = cupy.broadcast_arrays(base, exponent)

    # `.view` reinterprets the bits and is exact for both signs; `.astype`
    # would be a value conversion, and out-of-range signed conversion is
    # implementation-defined.
    accumulator = cupy.ascontiguousarray(base).view(unsigned).copy()
    remaining = cupy.ascontiguousarray(exponent).view(unsigned).copy()
    result = cupy.ones(accumulator.shape, dtype=unsigned)

    one = unsigned.type(1)
    largest = int(remaining.max()) if remaining.size else 0
    for _ in range(max(largest.bit_length(), 1)):
        odd = (remaining & one).astype(bool)
        result = cupy.where(odd, result * accumulator, result)
        accumulator = accumulator * accumulator
        remaining = remaining >> one
    return result.view(native)


def _ieee_pow(left, right):
    """IEEE 754 clause 9.2 ``pow`` over arrays (sections 12.2 and 12.3).

    CuPy follows the standard for almost every row of the table in section
    12.3.3. Two rows need correcting, and the correction is applied
    unconditionally because deciding whether it is needed would mean reading
    the operands back to the host:

    - ``(-0.0) ** y`` for a positive non-integral ``y`` must be ``+0.0``.
    - ``(-inf) ** y`` for a positive non-integral ``y`` must be ``+inf``.

    For a base of ``-0.0`` or ``-inf`` the standard's result depends only on
    the magnitude of the base and the parity of the exponent, so recomputing
    those elements from ``abs(base)`` gives the specified value. The mask is
    elementwise, so nothing is transferred and nothing synchronises.
    """
    result = cupy.power(left, right)
    signed_pole = cupy.signbit(left) & ((left == 0) | cupy.isinf(left))
    non_integral = right != cupy.trunc(right)
    return cupy.where(
        signed_pole & non_integral, cupy.power(cupy.abs(left), right), result
    )


def power(
    left: Any,
    right: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage in the declared dtype. Never declines."""
    if dtype.kind == "integer":
        # Native fixed-width exponentiation; see _integer_power. This is what
        # closes the gap where integer operands were declined and the Python
        # reference answered instead.
        native = cupy.dtype(dtype.name)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = _integer_power(
                left,
                right,
                native,
            )
        return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)

    # Floating exponentiation is non-trapping: every exceptional value in
    # section 12.3.3 is a result. Nothing here declines, so an infinity or a
    # NaN never sends the work to another backend, and the declared dtype is
    # preserved rather than widened to float64.
    if dtype.typecode == "f":
        # CuPy's generated binary32 code flushes subnormals, which section 5.4
        # forbids; the kernel below keeps them. The operands are handed over
        # untouched: an ElementwiseKernel broadcasts them itself, and routing a
        # scalar through cupy.asarray first would lose the sign of a negative
        # zero, which section 12.3.3 specifies.
        result = ieee32.apply("power", left, right)
        return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)

    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = _ieee_pow(left, right)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
