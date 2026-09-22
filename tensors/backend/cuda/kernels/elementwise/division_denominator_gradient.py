"""CuPy implementation of the division-denominator VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING, Any
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage

if TYPE_CHECKING:
    from tensors.dtype import DataType


def division_denominator_gradient(
    grad_values: Any,
    numerator_values: Any,
    denominator_values: Any,
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
    grad_shape: tuple[int, ...],
    numerator_shape: tuple[int, ...],
    denominator_shape: tuple[int, ...],
) -> Storage:
    """Calculate ``-grad * numerator / denominator**2``.

    ``direct`` is the expression evaluated as written, which is the specified
    result wherever floating point can carry it — including a zero divisor,
    where it gives the infinity or NaN of section 7.2.

    It loses the value only when ``denominator**2`` overflows or underflows
    although the true quotient is representable. ``stable`` recovers those in
    the logarithm, and the selection below picks it only for them. The
    operands are never read back to the host: the choice is made elementwise
    on the device, and the native arrays broadcast directly from their logical
    shapes.
    """
    upstream = grad_values
    values = numerator_values
    divisors = denominator_values
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
    return _arithmetic_storage(result, dtype=dtype, output_shape=output_shape)
