"""NumPy implementation of a power's second partial by its exponent, twice."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage
from tensors.backend.numpy.conversion import _operand

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def _ordinary(outer, upstream, bases, powers):
    """``outer * upstream * bases ** powers * log(bases) ** 2``.

    Valid where the base is strictly positive; the caller masks the rest in.
    The squared logarithm is never negative, so the sign is the sign of the
    two upstream gradients alone.
    """
    logarithm = numpy.log(bases)
    power_term = numpy.power(bases, powers)
    weight = outer * upstream
    direct = weight * power_term * logarithm * logarithm

    log_magnitude = (
        numpy.log(numpy.abs(outer))
        + numpy.log(numpy.abs(upstream))
        + 2.0 * numpy.log(numpy.abs(logarithm))
        + powers * logarithm
    )
    stable = numpy.copysign(numpy.exp(log_magnitude), weight)

    unsafe = (
        (power_term == 0.0)
        | ~numpy.isfinite(power_term)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    return numpy.where(unsafe, stable, direct), logarithm


def power_exponent_exponent_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Return the second partial by the exponent, weighted by both gradients.

    Nothing reads the host to decide a region (rule G3), no numerical
    condition raises (rule G2), and the gradient carries the *exponent's*
    declared dtype (rule G5).
    """
    try:
        outer_values = _operand(outer, outer.dtype)
        upstream = _operand(grad, grad.dtype)
        bases = _operand(base, base.dtype)
        powers = _operand(exponent, exponent.dtype)
    except (OverflowError, TypeError, ValueError):
        return None

    shape = (
        outer.shape.broadcast_with(grad.shape)
        .broadcast_with(base.shape)
        .broadcast_with(exponent.shape)
    )

    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result, logarithm = _ordinary(outer_values, upstream, bases, powers)

        zero_factor = (outer_values == 0.0) | (upstream == 0.0) | (logarithm == 0.0)
        result = numpy.where(zero_factor, 0.0, result)

        nan = numpy.float64("nan")
        zero_base = bases == 0.0
        # f(0, y) is zero for every y > 0, so it is constant in the exponent
        # and every derivative by the exponent is exactly zero.
        result = numpy.where(zero_base & (powers > 0.0), 0.0, result)
        result = numpy.where(zero_base & (powers <= 0.0), nan, result)
        result = numpy.where(bases < 0.0, nan, result)
        result = numpy.where(numpy.isnan(bases) | numpy.isnan(powers), nan, result)

    return _arithmetic_storage(result, dtype=exponent.dtype, output_shape=tuple(shape))
