"""NumPy implementation of the division-denominator VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import tensor_to_logical_array
from tensors.backend.numpy.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def division_denominator_gradient(
    grad: Tensor, numerator: Tensor, denominator: Tensor
) -> Storage | None:
    """Calculate ``-grad * numerator / denominator**2``.

    ``direct`` is the expression evaluated as written, which is the specified
    result wherever floating point can carry it — including a zero divisor,
    where it gives the infinity or NaN of section 7.2.

    It loses the value only when ``denominator**2`` overflows or underflows
    although the true quotient is representable. ``stable`` recovers those in
    the logarithm, and the selection below picks it only for them. The
    operands are never read back to the host: the choice is made elementwise
    on the device.
    """
    native = numpy.dtype(grad.dtype.name)
    upstream = tensor_to_logical_array(grad).astype(native, copy=False)
    values = tensor_to_logical_array(numerator).astype(native, copy=False)
    divisors = tensor_to_logical_array(denominator).astype(native, copy=False)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        squares = numpy.square(divisors)
        direct = -upstream * values / squares

        # A zero divisor, a zero factor or a non-finite operand all make
        # ``direct`` the specified answer, so only the remaining elements can
        # need rescuing.
        ordinary = (
            (divisors != 0.0)
            & (upstream != 0.0)
            & (values != 0.0)
            & numpy.isfinite(divisors)
            & numpy.isfinite(upstream)
            & numpy.isfinite(values)
        )
        lost = ordinary & (
            (squares == 0.0)
            | ~numpy.isfinite(squares)
            | (direct == 0.0)
            | ~numpy.isfinite(direct)
        )

        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + numpy.log(numpy.abs(values))
            - 2.0 * numpy.log(numpy.abs(divisors))
        )
        sign = numpy.where(numpy.signbit(upstream) ^ numpy.signbit(values), 1.0, -1.0)
        stable = sign * numpy.exp(log_magnitude)
        result = numpy.where(lost, stable, direct)
    return _arithmetic_storage(result, dtype=grad.dtype, output_shape=grad.shape)
