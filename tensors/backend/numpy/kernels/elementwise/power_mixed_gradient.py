"""NumPy implementation of a power's mixed second partial."""

from __future__ import annotations
import numpy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.numpy.conversion import _errstate
from tensors.backend.numpy.conversion import _arithmetic_storage
from tensors.backend.numpy.conversion import _operand

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _ordinary(outer, upstream, bases, powers):
    """``outer * upstream * bases ** (powers - 1) * (1 + powers * log(bases))``.

    Valid where the base is strictly positive, which is where ``log`` is
    real; the caller masks every other region in. The direct product is used
    where it is representable and a logarithmic form carries the magnitude
    where the power would overflow or vanish.
    """
    logarithm = numpy.log(bases)
    coefficient = 1.0 + powers * logarithm
    power_term = numpy.power(bases, powers - 1.0)
    weight = outer * upstream * coefficient
    direct = weight * power_term

    log_magnitude = (
        numpy.log(numpy.abs(outer))
        + numpy.log(numpy.abs(upstream))
        + numpy.log(numpy.abs(coefficient))
        + (powers - 1.0) * logarithm
    )
    stable = numpy.copysign(numpy.exp(log_magnitude), weight)

    unsafe = (
        (power_term == 0.0)
        | ~numpy.isfinite(power_term)
        | (direct == 0.0)
        | ~numpy.isfinite(direct)
    )
    return numpy.where(unsafe, stable, direct), coefficient


def power_mixed_gradient(
    outer: Tensor,
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
    *,
    dtype: DataType,
) -> Storage | None:
    """Return the mixed second partial, weighted by both upstream gradients.

    The caller names the dtype because this one partial serves two gradients:
    it is the exponent's when the base VJP is differentiated and the base's
    when the exponent VJP is, and rule G5 gives each the dtype of the operand
    it belongs to.

    Nothing reads the host to decide a region (rule G3) and no numerical
    condition raises (rule G2).
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
        result, coefficient = _ordinary(outer_values, upstream, bases, powers)

        zero_factor = (outer_values == 0.0) | (upstream == 0.0) | (coefficient == 0.0)
        result = numpy.where(zero_factor, 0.0, result)

        nan = numpy.float64("nan")
        zero_base = bases == 0.0
        # The base gradient is zero for every exponent above one at a zero
        # base, so it is constant in the exponent and the mixed partial is
        # exactly zero. At and below one no derivative exists.
        result = numpy.where(zero_base & (powers > 1.0), 0.0, result)
        result = numpy.where(zero_base & (powers <= 1.0), nan, result)
        # ``ln x`` is undefined for a negative base, at every exponent.
        result = numpy.where(bases < 0.0, nan, result)
        result = numpy.where(numpy.isnan(bases) | numpy.isnan(powers), nan, result)

    return _arithmetic_storage(result, dtype=dtype, output_shape=tuple(shape))
