"""CuPy implementation of a power's second partial by its base, twice."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.kernels.arithmetic import ieee32

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def _ordinary(outer, upstream, bases, powers):
    """``outer * upstream * powers * (powers - 1) * bases ** (powers - 2)``.

    The direct product is used where it is representable, and a logarithmic
    form takes over where an intermediate would overflow or underflow to
    nothing — the same strategy the first-order kernels use, because the same
    thing goes wrong: the power leaves the range while the whole expression
    stays inside it.

    The sign is reconstructed rather than recovered from the magnitude. A
    negative base raised to an odd integral power contributes a sign the
    logarithm of the magnitude cannot carry.
    """
    power_term = cupy.power(bases, powers - 2.0)
    coefficient = outer * upstream * powers * (powers - 1.0)
    direct = coefficient * power_term

    log_magnitude = (
        cupy.log(cupy.abs(outer))
        + cupy.log(cupy.abs(upstream))
        + cupy.log(cupy.abs(powers))
        + cupy.log(cupy.abs(powers - 1.0))
        + (powers - 2.0) * cupy.log(cupy.abs(bases))
    )
    odd_negative = (bases < 0.0) & (cupy.abs(cupy.fmod(powers - 2.0, 2.0)) == 1.0)
    sign = cupy.where(odd_negative, -coefficient, coefficient)
    stable = cupy.copysign(cupy.exp(log_magnitude), sign)

    unsafe = (
        (power_term == 0.0)
        | ~cupy.isfinite(power_term)
        | (direct == 0.0)
        | ~cupy.isfinite(direct)
    )
    return cupy.where(unsafe, stable, direct)


def power_base_base_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Return the second partial by the base, weighted by both upstream gradients.

    Nothing is read back to the host to decide a region (rule G3) and no
    numerical condition raises (rule G2). The gradient carries the *base's*
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
        result = _ordinary(outer_values, upstream, bases, powers)

        # An exactly zero factor is an exact zero, whatever the power did.
        # This is what makes the second derivative of a constant and of a
        # linear function zero rather than the ``0 * inf`` of the formula.
        zero_factor = (
            (outer_values == 0.0)
            | (upstream == 0.0)
            | (powers == 0.0)
            | (powers == 1.0)
        )
        result = cupy.where(zero_factor, 0.0, result)

        # A negative base with a non-integral exponent has no real forward
        # value, so it has no derivative of any order either.
        nan = cupy.float64("nan")
        integral = powers == cupy.trunc(powers)
        result = cupy.where((bases < 0.0) & ~integral, nan, result)
        result = cupy.where(cupy.isnan(bases) | cupy.isnan(powers), nan, result)

    if base.dtype.typecode == "f":
        # Narrow through PTX: astype flushes a binary32 subnormal,
        # which section 5.4 forbids. See ieee32._build_narrow.
        result = ieee32.narrow(cupy.asarray(result, dtype=cupy.float64))
    return _arithmetic_storage(result, dtype=base.dtype, output_shape=tuple(shape))
