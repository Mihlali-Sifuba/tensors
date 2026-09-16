"""NumPy implementation of the power-exponent VJP."""

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


def power_exponent_gradient(
    grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Calculate the power gradient with respect to its exponent when safe."""
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
        logarithm = numpy.log(bases)
        outputs = numpy.power(bases, powers)
        direct = upstream * outputs * logarithm
        zero = (upstream == 0.0) | (logarithm == 0.0)
        log_magnitude = (
            numpy.log(numpy.abs(upstream))
            + powers * logarithm
            + numpy.log(numpy.abs(logarithm))
        )
        stable = numpy.where(
            zero, 0.0, numpy.copysign(numpy.exp(log_magnitude), upstream * logarithm)
        )
    unsafe = ~zero & (
        (outputs == 0.0)
        | ~numpy.isfinite(outputs)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    result = numpy.where(unsafe, stable, direct)
    return _storage(result, dtype=grad.dtype, output_shape=grad.shape)
