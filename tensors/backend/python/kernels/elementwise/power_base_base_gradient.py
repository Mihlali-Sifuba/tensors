"""Reference the second partial of a power by its base, twice."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
from tensors.backend.python.storage import PythonStorage
from tensors.tensor import Tensor
from tensors.utils.power_gradients import is_integral, power_product

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def _base_base_value(
    outer: float, upstream: float, base: float, exponent: float
) -> float:
    """Return ``outer * upstream * d2(base ** exponent)/d(base)2``.

    The formula is ``e (e - 1) b ** (e - 2)``, evaluated by
    :func:`~tensors.utils.power_gradients.power_product` so that a power
    leaving the representable range does not discard a result that stays
    inside it.

    An exponent of zero or one makes a factor exactly zero, and the
    short-circuit in that function is what returns an exact zero for the
    second derivative of a constant and of a linear function, rather than
    the ``0 * inf`` the unguarded formula produces at a zero base.

    A negative base with a non-integral exponent has no real forward value,
    so no derivative of any order exists there and the answer is NaN (rule
    G2). With an integral exponent ``e - 2`` is integral too, the power is
    real, and the ordinary formula holds.
    """
    if base != base or exponent != exponent:
        return math.nan
    if base < 0.0 and not is_integral(exponent):
        return math.nan
    return power_product(
        [outer, upstream, exponent, exponent - 1.0], base, exponent - 2.0
    )


def power_base_base_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
) -> Storage | None:
    """Scale both upstream gradients by ``exponent * (exponent - 1) * base ** (exponent - 2)``."""
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
        _base_base_value(
            float(outer_value), float(upstream), float(base_value), float(power)
        )
        for outer_value, upstream, base_value, power in zip(
            outer._data, grad._data, base._data, exponent._data
        )
    ]
    # Rule G5: the gradient carries the *base's* declared dtype.
    return PythonStorage.from_values(values, base.dtype)
