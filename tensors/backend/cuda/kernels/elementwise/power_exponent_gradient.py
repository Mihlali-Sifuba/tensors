"""CuPy implementation of the power-exponent VJP (section 12.7)."""

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
    """``upstream * bases ** powers * log(bases)`` without range loss.

    Valid only where ``bases > 0``; every other row of section 12.7.2 gives
    NaN for this derivative, and the caller masks them in. The direct product
    is used where it is representable and a logarithmic form takes over where
    an intermediate would overflow or vanish, as before.
    """
    logarithm = cupy.log(bases)
    outputs = cupy.power(bases, powers)
    direct = upstream * outputs * logarithm

    zero = (upstream == 0.0) | (logarithm == 0.0)
    log_magnitude = (
        cupy.log(cupy.abs(upstream))
        + powers * logarithm
        + cupy.log(cupy.abs(logarithm))
    )
    stable = cupy.where(
        zero, 0.0, cupy.copysign(cupy.exp(log_magnitude), upstream * logarithm)
    )

    unsafe = ~zero & (
        (outputs == 0.0)
        | ~cupy.isfinite(outputs)
        | (direct == 0.0)
        | ~cupy.isfinite(direct)
    )
    return cupy.where(unsafe, stable, direct)


def power_exponent_gradient(
    grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Return the VJP with respect to the exponent, by the same table.

    A zero base is not uniformly zero, as the previous form assumed by
    declining: it is zero only for a strictly positive exponent. A negative
    base has no exponent derivative at any exponent, because ``ln x`` is
    undefined there, and that is NaN rather than a reason to decline.

    Nothing reads the host to decide a region (rule G3) and the gradient
    carries the *exponent's* declared dtype (rule G5).
    """
    try:
        upstream = _operand(grad, grad.dtype)
        bases = _operand(base, base.dtype)
        powers = _operand(exponent, exponent.dtype)
    except (OverflowError, TypeError, ValueError):
        return None

    with _errstate(divide="ignore", over="ignore", under="ignore", invalid="ignore"):
        result = _ordinary(upstream, bases, powers)

        zero_base = bases == 0.0
        nan = cupy.float64("nan")

        # x = 0, y > 0 -> 0. f(0, y) = 0 for every y > 0, so f is constant in
        # y and the derivative is exactly zero; the formula's 0 * (-inf) is a
        # degenerate encoding of a derivative that genuinely exists.
        result = cupy.where(zero_base & (powers > 0.0), upstream * 0.0, result)
        # x = 0, y = 0 -> NaN: f(0, y) is discontinuous there.
        # x = 0, y < 0 -> NaN.
        result = cupy.where(zero_base & (powers <= 0.0), upstream * nan, result)
        # x < 0 -> NaN for every exponent, integral or not.
        result = cupy.where(bases < 0.0, upstream * nan, result)

    if exponent.dtype.typecode == "f":
        # Narrow through PTX: astype flushes a binary32 subnormal,
        # which section 5.4 forbids. See _ieee32._build_narrow.
        result = _ieee32.narrow(cupy.asarray(result, dtype=cupy.float64))
    return _arithmetic_storage(result, dtype=exponent.dtype, output_shape=grad.shape)
