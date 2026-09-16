"""CuPy implementation of the division-denominator VJP."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _finite_operands
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.conversion import _storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def division_denominator_gradient(
    grad: Tensor, numerator: Tensor, denominator: Tensor
) -> Storage | None:
    """Calculate ``-grad * numerator / denominator**2`` when range-safe."""
    try:
        upstream = _operand(grad, grad.dtype)
        values = _operand(numerator, grad.dtype)
        divisors = _operand(denominator, grad.dtype)
    except (OverflowError, TypeError, ValueError):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        squares = cupy.square(divisors)
    finite_inputs = _finite_operands(upstream, values, divisors)
    if not finite_inputs or bool(cupy.any(divisors == 0.0)):
        return None
    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        direct = -upstream * values / squares
        zero = (upstream == 0.0) | (values == 0.0)
        log_magnitude = (
            cupy.log(cupy.abs(upstream))
            + cupy.log(cupy.abs(values))
            - 2.0 * cupy.log(cupy.abs(divisors))
        )
        sign = cupy.where(cupy.signbit(upstream) ^ cupy.signbit(values), 1.0, -1.0)
        stable = cupy.where(zero, 0.0, sign * cupy.exp(log_magnitude))
    unsafe = (
        (squares == 0.0)
        | ~cupy.isfinite(squares)
        | ~zero & ((direct == 0.0) | ~cupy.isfinite(direct))
    )
    result = cupy.where(unsafe, stable, direct)
    return _storage(result, dtype=grad.dtype, output_shape=grad.shape)
