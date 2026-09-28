"""CuPy implementation of a power's second partial by its exponent, twice."""

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
    """``outer * upstream * bases ** powers * log(bases) ** 2``.

    Valid where the base is strictly positive; the caller masks the rest in.
    The squared logarithm is never negative, so the sign is the sign of the
    two upstream gradients alone.
    """
    logarithm = cupy.log(bases)
    power_term = cupy.power(bases, powers)
    weight = outer * upstream
    direct = weight * power_term * logarithm * logarithm

    log_magnitude = (
        cupy.log(cupy.abs(outer))
        + cupy.log(cupy.abs(upstream))
        + 2.0 * cupy.log(cupy.abs(logarithm))
        + powers * logarithm
    )
    stable = cupy.copysign(cupy.exp(log_magnitude), weight)

    unsafe = (
        (power_term == 0.0)
        | ~cupy.isfinite(power_term)
        | (direct == 0.0)
        | ~cupy.isfinite(direct)
    )
    return cupy.where(unsafe, stable, direct), logarithm


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
        result = cupy.where(zero_factor, 0.0, result)

        nan = cupy.float64("nan")
        zero_base = bases == 0.0
        # f(0, y) is zero for every y > 0, so it is constant in the exponent
        # and every derivative by the exponent is exactly zero.
        result = cupy.where(zero_base & (powers > 0.0), 0.0, result)
        result = cupy.where(zero_base & (powers <= 0.0), nan, result)
        result = cupy.where(bases < 0.0, nan, result)
        result = cupy.where(cupy.isnan(bases) | cupy.isnan(powers), nan, result)

    if exponent.dtype.typecode == "f":
        # Narrow through PTX: astype flushes a binary32 subnormal,
        # which section 5.4 forbids. See ieee32._build_narrow.
        result = ieee32.narrow(cupy.asarray(result, dtype=cupy.float64))
    return _arithmetic_storage(result, dtype=exponent.dtype, output_shape=tuple(shape))
