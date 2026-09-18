"""CuPy implementation of the power-base VJP (section 12.7)."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.kernels.arithmetic import _ieee32

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def _ordinary(upstream, bases, powers):
    """``upstream * powers * bases ** (powers - 1)`` without range loss.

    This is the section 12.7.2 formula, for the rows where it is valid. The
    direct product is used where it is representable, and a logarithmic form
    takes over where an intermediate would overflow or underflow to nothing —
    the computation that was already here, kept because it is still correct.

    The logarithmic form now carries its own sign reconstruction. It used to
    take ``log(bases)``, which is NaN for a negative base; that never showed
    because the kernel declined on negative bases before reaching it. A
    negative base with an integral exponent is an ordinary row of the table,
    so the magnitude comes from ``log|bases|`` and the sign from the parity of
    ``powers - 1``.
    """
    power_term = cupy.power(bases, powers - 1.0)
    direct = upstream * powers * power_term

    zero = (upstream == 0.0) | (powers == 0.0)
    log_magnitude = (
        cupy.log(cupy.abs(upstream))
        + cupy.log(cupy.abs(powers))
        + (powers - 1.0) * cupy.log(cupy.abs(bases))
    )
    # sign(upstream * powers), flipped when a negative base is raised to an
    # odd integral power.
    odd_negative = (bases < 0.0) & (cupy.abs(cupy.fmod(powers - 1.0, 2.0)) == 1.0)
    sign = cupy.where(odd_negative, -(upstream * powers), upstream * powers)
    stable = cupy.where(zero, 0.0, cupy.copysign(cupy.exp(log_magnitude), sign))

    unsafe = ~zero & (
        (power_term == 0.0)
        | ~cupy.isfinite(power_term)
        | (direct == 0.0)
        | ~cupy.isfinite(direct)
    )
    return cupy.where(unsafe, stable, direct)


def power_base_gradient(grad: Tensor, base: Tensor, exponent: Tensor) -> Storage | None:
    """Return the VJP with respect to the base, by the section 12.7.2 table.

    Every row is applied as a mask. Nothing is read back to the host to decide
    a region (rule G3), nothing declines because of an operand's value, and no
    numerical condition raises (rule G2). The gradient carries the *base's*
    declared dtype (rule G5).
    """
    try:
        upstream = _operand(grad, grad.dtype)
        bases = _operand(base, base.dtype)
        powers = _operand(exponent, exponent.dtype)
    except (OverflowError, TypeError, ValueError):
        return None

    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = _ordinary(upstream, bases, powers)

        # The classified rows. Each is the derivative, so each is scaled by
        # the upstream gradient; 0 * inf and 0 * NaN are left to IEEE, since
        # section 12.7 approves no convention for those products.
        zero_base = bases == 0.0
        integral = powers == cupy.trunc(powers)
        nan = cupy.float64("nan")
        infinity = cupy.float64("inf")

        # x = 0, y = 0 -> 0, and x = 0, y > 1 -> 0. f is constant in x there.
        result = cupy.where(
            zero_base & ((powers == 0.0) | (powers > 1.0)), upstream * 0.0, result
        )
        # x = 0, y = 1 -> 1
        result = cupy.where(zero_base & (powers == 1.0), upstream * 1.0, result)
        # x = 0, 0 < y < 1 -> +inf, an approved convention for a one-sided
        # infinite slope, not a finite derivative.
        result = cupy.where(
            zero_base & (powers > 0.0) & (powers < 1.0), upstream * infinity, result
        )
        # x = 0, y < 0 -> NaN. The forward result is an infinity, which does
        # not establish that a derivative exists at the evaluation point.
        result = cupy.where(zero_base & (powers < 0.0), upstream * nan, result)
        # x < 0 with a non-integral exponent has no real forward value either.
        result = cupy.where((bases < 0.0) & ~integral, upstream * nan, result)

    if base.dtype.typecode == "f":
        # Narrow through PTX: astype flushes a binary32 subnormal,
        # which section 5.4 forbids. See _ieee32._build_narrow.
        result = _ieee32.narrow(cupy.asarray(result, dtype=cupy.float64))
    return _arithmetic_storage(result, dtype=base.dtype, output_shape=grad.shape)
