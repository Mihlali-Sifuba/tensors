"""NumPy implementation of the power-base VJP."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _finite_operands
from tensors.backend.numpy.conversion import _operand
from tensors.backend.numpy.conversion import _storage

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def power_base_gradient(grad: Tensor, base: Tensor, exponent: Tensor) -> Storage | None:
    """Calculate the power gradient with respect to its base when safe."""
    try:
        upstream = _operand(grad, grad.dtype)
        bases = _operand(base, grad.dtype)
        powers = _operand(exponent, grad.dtype)
    except (OverflowError, TypeError, ValueError):
        return None
    if not _finite_operands(upstream, bases, powers):
        return None
    if bool(numpy.any(bases <= 0.0)):
        return None
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        power_term = numpy.power(bases, powers - 1.0)
        direct = upstream * powers * power_term
        zero = (upstream == 0.0) | (powers == 0.0)
        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + numpy.log(numpy.abs(powers))
            + (powers - 1.0) * numpy.log(bases)
        )
        stable = numpy.where(
            zero, 0.0, numpy.copysign(numpy.exp(log_magnitude), upstream * powers)
        )
    unsafe = ~zero & (
        (power_term == 0.0)
        | ~numpy.isfinite(power_term)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    result = numpy.where(unsafe, stable, direct)
    return _storage(result, dtype=grad.dtype, output_shape=grad.shape)
