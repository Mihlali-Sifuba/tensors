"""NumPy implementation of exponentiation."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_operand
from tensors.backend.numpy.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors._typing import Scalar
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


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
    unsigned = numpy.dtype(_UNSIGNED_OF_WIDTH[native.itemsize])

    base = numpy.ascontiguousarray(numpy.asarray(left, dtype=native))
    exponent = numpy.ascontiguousarray(numpy.asarray(right, dtype=native))
    base, exponent = numpy.broadcast_arrays(base, exponent)

    # `.view` reinterprets the bits and is exact for both signs; `.astype`
    # would be a value conversion, and out-of-range signed conversion is
    # implementation-defined.
    accumulator = numpy.ascontiguousarray(base).view(unsigned).copy()
    remaining = numpy.ascontiguousarray(exponent).view(unsigned).copy()
    result = numpy.ones(accumulator.shape, dtype=unsigned)

    one = unsigned.type(1)
    largest = int(remaining.max()) if remaining.size else 0
    for _ in range(max(largest.bit_length(), 1)):
        odd = (remaining & one).astype(bool)
        result = numpy.where(odd, result * accumulator, result)
        accumulator = accumulator * accumulator
        remaining = remaining >> one
    return result.view(native)


def _ieee_pow(left, right):
    """IEEE 754 clause 9.2 ``pow`` over arrays (sections 12.2 and 12.3).

    NumPy follows the standard for almost every row of the table in section
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
    result = numpy.power(left, right)
    signed_pole = numpy.signbit(left) & ((left == 0) | numpy.isinf(left))
    non_integral = right != numpy.trunc(right)
    return numpy.where(
        signed_pole & non_integral, numpy.power(numpy.abs(left), right), result
    )


def _binary32_pow(left, right):
    """Binary32 exponentiation, evaluated in binary64 and rounded once.

    NumPy evaluates a binary32 power in binary32, and its ``exp(y log x)``
    loses the bottom of the subnormal range: the smallest binary32 subnormal
    raised to 1.0000001192092896 returns zero where the correctly rounded
    result is the subnormal itself. That is a section 5.4 gradual-underflow
    failure and a section 12.6.2 conformance failure at once — the result is
    not a few ULP out, it is the wrong class of value.

    Widening is the same strategy the CUDA binary32 kernel already uses, and
    for the same reason. It is sound rather than merely better: a binary64
    ``pow`` carries a relative error of order 2**-52, which is about 2**-28 of
    a binary32 ULP, so narrowing the binary64 result reproduces the correctly
    rounded binary32 value except where the true value lies within that
    distance of a binary32 rounding boundary, and there it is one ULP out.
    Both are far inside the 4 ULP of section 12.6.2.

    The special values of section 12.3.3 are unaffected: widening and
    narrowing carry a signed zero, a signed infinity and a NaN across
    unchanged, and :func:`_ieee_pow` still applies the two rows NumPy gets
    wrong.
    """
    wide = _ieee_pow(
        numpy.asarray(left, dtype=numpy.float64),
        numpy.asarray(right, dtype=numpy.float64),
    )
    return wide.astype(numpy.float32, copy=False)


def power(
    left: Tensor | Scalar,
    right: Tensor | Scalar,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return native storage in the declared dtype. Never declines."""
    if dtype.kind == "integer":
        # Native fixed-width exponentiation; see _integer_power. This is what
        # closes the gap where integer operands were declined and the Python
        # reference answered instead.
        native = numpy.dtype(dtype.name)
        with _errstate(over="ignore", under="ignore", invalid="ignore"):
            result = _integer_power(
                _arithmetic_operand(left, dtype),
                _arithmetic_operand(right, dtype),
                native,
            )
        return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)

    # Floating exponentiation is non-trapping: every exceptional value in
    # section 12.3.3 is a result. Nothing here declines, so an infinity or a
    # NaN never sends the work to another backend, and the declared dtype is
    # preserved rather than widened to float64.
    left_array = _arithmetic_operand(left, dtype)
    right_array = _arithmetic_operand(right, dtype)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        if dtype.typecode == "f":
            result = _binary32_pow(left_array, right_array)
        else:
            result = _ieee_pow(left_array, right_array)
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
