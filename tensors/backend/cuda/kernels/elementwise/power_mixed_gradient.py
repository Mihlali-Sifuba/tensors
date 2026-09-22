"""CuPy implementation of a power's mixed second partial."""

from __future__ import annotations
import cupy
from typing import TYPE_CHECKING
from tensors.backend.storage import Storage
from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _arithmetic_storage
from tensors.backend.cuda.conversion import _operand
from tensors.backend.cuda.kernels.arithmetic import ieee32

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
    logarithm = cupy.log(bases)
    coefficient = 1.0 + powers * logarithm
    power_term = cupy.power(bases, powers - 1.0)
    weight = outer * upstream * coefficient
    direct = weight * power_term

    log_magnitude = (
        cupy.log(cupy.abs(outer))
        + cupy.log(cupy.abs(upstream))
        + cupy.log(cupy.abs(coefficient))
        + (powers - 1.0) * logarithm
    )
    stable = cupy.copysign(cupy.exp(log_magnitude), weight)

    unsafe = (
        (power_term == 0.0)
        | ~cupy.isfinite(power_term)
        | (direct == 0.0)
        | ~cupy.isfinite(direct)
    )
    return cupy.where(unsafe, stable, direct), coefficient


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
        result = cupy.where(zero_factor, 0.0, result)

        nan = cupy.float64("nan")
        zero_base = bases == 0.0
        # The base gradient is zero for every exponent above one at a zero
        # base, so it is constant in the exponent and the mixed partial is
        # exactly zero. At and below one no derivative exists.
        result = cupy.where(zero_base & (powers > 1.0), 0.0, result)
        result = cupy.where(zero_base & (powers <= 1.0), nan, result)
        # ``ln x`` is undefined for a negative base, at every exponent.
        result = cupy.where(bases < 0.0, nan, result)
        result = cupy.where(cupy.isnan(bases) | cupy.isnan(powers), nan, result)

    if dtype.typecode == "f":
        # Narrow through PTX: astype flushes a binary32 subnormal,
        # which section 5.4 forbids. See ieee32._build_narrow.
        result = ieee32.narrow(cupy.asarray(result, dtype=cupy.float64))
    return _arithmetic_storage(result, dtype=dtype, output_shape=tuple(shape))
