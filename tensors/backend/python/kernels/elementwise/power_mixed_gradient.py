"""Reference the mixed second partial of a power, by base and by exponent."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING
from tensors.backend.python.storage import PythonStorage
from tensors.dtype import DataType
from tensors.tensor import Tensor
from tensors.utils.power_gradients import power_product

if TYPE_CHECKING:
    from tensors.backend.storage import Storage


def _mixed_value(outer: float, upstream: float, base: float, exponent: float) -> float:
    """Return ``outer * upstream * d2(base ** exponent)/d(base)d(exponent)``.

    Differentiating the base gradient by the exponent and the exponent
    gradient by the base give the same mixed second partial, so both of
    power's vector-Jacobian products reach this one statement of it and
    neither carries a copy that could drift from the other.

    The formula is ``b ** (e - 1) * (1 + e ln b)``, and ``ln b`` is real
    only for a positive base.

    At a zero base the base gradient ``e b ** (e - 1)`` is zero for every
    exponent above one, so it is constant in the exponent there and the
    mixed partial is exactly zero. At and below one it is not, and no
    derivative exists: NaN records that (rule G2), where this computation
    previously raised and in raising discarded the other partial as well.

    For a negative base the exponent derivative being differentiated does
    not exist at any exponent, because ``ln x`` is undefined there, so
    neither does the mixed partial.
    """
    if base != base or exponent != exponent:
        return math.nan
    if base < 0.0:
        return math.nan
    if base == 0.0:
        return 0.0 if exponent > 1.0 else math.nan
    coefficient = math.fsum([1.0, exponent * math.log(base)])
    return power_product([outer, upstream, coefficient], base, exponent - 1.0)


def power_mixed_gradient(
    outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor, *, dtype: DataType
) -> Storage | None:
    """Scale both upstream gradients by ``base ** (exponent - 1) * (1 + exponent * log(base))``.

    The caller names the dtype because this one partial serves two gradients:
    it is the exponent's gradient when the base VJP is differentiated and the
    base's gradient when the exponent VJP is, and rule G5 gives each the dtype
    of the operand it belongs to.
    """
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
        _mixed_value(
            float(outer_value), float(upstream), float(base_value), float(power)
        )
        for outer_value, upstream, base_value, power in zip(
            outer._data, grad._data, base._data, exponent._data
        )
    ]
    return PythonStorage.from_values(values, dtype)
