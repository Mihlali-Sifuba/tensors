"""CuPy implementation of the division-denominator VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_operand
from tensors.backend.cuda.conversion import _arithmetic_storage

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
    upstream = _arithmetic_operand(grad, grad.dtype)
    values = _arithmetic_operand(numerator, grad.dtype)
    divisors = _arithmetic_operand(denominator, grad.dtype)
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        squares = cupy.square(divisors)
        direct = -upstream * values / squares

        # A zero divisor, a zero factor or a non-finite operand all make
        # ``direct`` the specified answer, so only the remaining elements can
        # need rescuing.
        ordinary = (
            (divisors != 0.0)
            & (upstream != 0.0)
            & (values != 0.0)
            & cupy.isfinite(divisors)
            & cupy.isfinite(upstream)
            & cupy.isfinite(values)
        )
        lost = ordinary & (
            (squares == 0.0)
            | ~cupy.isfinite(squares)
            | (direct == 0.0)
            | ~cupy.isfinite(direct)
        )

        log_magnitude = (
            cupy.log(cupy.abs(upstream))
            + cupy.log(cupy.abs(values))
            - 2.0 * cupy.log(cupy.abs(divisors))
        )
        sign = cupy.where(cupy.signbit(upstream) ^ cupy.signbit(values), 1.0, -1.0)
        stable = sign * cupy.exp(log_magnitude)
        result = cupy.where(lost, stable, direct)
    return _arithmetic_storage(result, dtype=grad.dtype, output_shape=grad.shape)
