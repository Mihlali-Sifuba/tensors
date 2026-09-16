"""Reference the RMSprop parameter update for the Python backend."""

from __future__ import annotations
import math
from tensors.backend.python.storage import PythonStorage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.backend.storage import Storage
    from tensors.tensor import Tensor


def _stable_weighted_sum(
    left_weight: float, left: float, right_weight: float, right: float
) -> float:
    """Return a two-term weighted sum with reliable cancellation."""
    from tensors.utils.summation import stable_float_sum

    return stable_float_sum([left_weight * left, right_weight * right])


def _scaled_second_moment(
    scale: float, scaled: float, gradient: float, decay: float
) -> tuple[float, float]:
    """Update a squared average without explicitly squaring a huge value."""
    new_scale = max(scale, abs(gradient))
    if new_scale == 0.0:
        return (0.0, 0.0)
    previous_ratio = scale / new_scale
    gradient_ratio = abs(gradient) / new_scale
    new_scaled = (
        decay * scaled * previous_ratio * previous_ratio
        + (1.0 - decay) * gradient_ratio * gradient_ratio
    )
    return (new_scale, new_scaled)


def _visible_second_moment(scale: float, scaled: float) -> float:
    """Materialize a second moment for inspection when it fits in a float."""
    if scale == 0.0 or scaled == 0.0:
        return 0.0
    try:
        return _product_quotient([scale, scale, scaled], [1.0])
    except OverflowError:
        return math.inf


def _product_quotient(numerators: list[float], denominators: list[float]) -> float:
    """Evaluate a finite product quotient as one exact binary ratio."""
    numerator = 1
    denominator = 1
    for value in numerators:
        value_numerator, value_denominator = float(value).as_integer_ratio()
        numerator *= value_numerator
        denominator *= value_denominator
    for value in denominators:
        value_numerator, value_denominator = float(value).as_integer_ratio()
        numerator *= value_denominator
        denominator *= value_numerator
    try:
        return numerator / denominator
    except OverflowError:
        return math.inf if numerator * denominator > 0 else -math.inf


def rmsprop_update(
    parameter: Tensor,
    gradient: Tensor,
    scale: Tensor,
    scaled: Tensor,
    *,
    rho: float,
    learning_rate: float,
    epsilon: float,
) -> tuple[Storage, Storage, Storage] | None:
    """Apply one RMSprop step to a parameter."""
    new_scales = []
    new_scaled_values = []
    parameter_values = []
    for parameter_value, gradient_value, scale, scaled in zip(
        parameter._data, gradient._data, scale._data, scaled._data
    ):
        gradient_value = float(gradient_value)
        new_scale, new_scaled = _scaled_second_moment(
            float(scale), float(scaled), gradient_value, rho
        )
        root_moment = new_scale * math.sqrt(new_scaled)
        update = learning_rate * (gradient_value / (root_moment + epsilon))
        new_scales.append(new_scale)
        new_scaled_values.append(new_scaled)
        parameter_values.append(float(parameter_value) - update)
    return (
        PythonStorage.from_values(parameter_values, parameter.dtype),
        PythonStorage.from_values(new_scales, gradient.dtype),
        PythonStorage.from_values(new_scaled_values, gradient.dtype),
    )
