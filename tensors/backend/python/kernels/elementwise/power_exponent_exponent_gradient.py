"""Reference the second partial of a power by its exponent, twice."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
from tensors.backend.python.storage import PythonStorage
from tensors.tensor import Tensor
from tensors.utils.power_gradients import power_product

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def _exponent_exponent_value(
    outer: float, upstream: float, base: float, exponent: float
) -> float:
    """Return ``outer * upstream * d2(base ** exponent)/d(exponent)2``.

    The formula is ``b ** e * (ln b) ** 2``, and ``ln b`` is real only for a
    positive base.

    At a zero base with a positive exponent the forward value is zero for
    every exponent, so it is constant in the exponent and every derivative
    by the exponent is exactly zero. At an exponent of zero or below the
    forward value is discontinuous or infinite and no derivative exists, so
    the answer is NaN (rule G2).

    For a negative base ``ln x`` is undefined, so the first exponent
    derivative does not exist and neither does the second.
    """
    if base != base or exponent != exponent:
        return math.nan
    if base < 0.0:
        return math.nan
    if base == 0.0:
        return 0.0 if exponent > 0.0 else math.nan
    logarithm = math.log(base)
    return power_product([outer, upstream, logarithm, logarithm], base, exponent)


def power_exponent_exponent_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Scale both upstream gradients by ``base ** exponent * log(base) ** 2``."""
    from tensors.utils.broadcasting import broadcast_to

    shape = (
        outer.shape.broadcast_with(grad.shape)
        .broadcast_with(base.shape)
        .broadcast_with(exponent.shape)
    )
    outer, grad, base, exponent = (
        broadcast_to(value, shape) for value in (outer, grad, base, exponent)
    )
    values = [
        _exponent_exponent_value(
            float(outer_value), float(upstream), float(base_value), float(power)
        )
        for outer_value, upstream, base_value, power in zip(
            outer._data, grad._data, base._data, exponent._data
        )
    ]
    # Rule G5: the gradient carries the *exponent's* declared dtype.
    return PythonStorage.from_values(values, exponent.dtype)
