"""Reference summation for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensors.tensor import Tensor
import builtins
import math
from tensors.math._reduction import reduction_groups


def _stable_float_sum(values: list[float]) -> float:
    """Sum floats accurately even when a temporary partial sum overflows."""
    if any((math.isnan(value) for value in values)):
        return math.nan
    has_positive_infinity = math.inf in values
    has_negative_infinity = -math.inf in values
    if has_positive_infinity and has_negative_infinity:
        return math.nan
    if has_positive_infinity:
        return math.inf
    if has_negative_infinity:
        return -math.inf
    try:
        return math.fsum(values)
    except OverflowError:
        return _sum_exact_ratios([value.as_integer_ratio() for value in values])


def _sum_exact_ratios(ratios: list[tuple[int, int]], *, divisor: int = 1) -> float:
    """Convert an exact sum of binary ratios, optionally divided, to a float."""
    denominator = max((item[1] for item in ratios), default=1)
    numerator = builtins.sum(
        (
            item_numerator * (denominator // item_denominator)
            for item_numerator, item_denominator in ratios
        )
    )
    try:
        return numerator / (denominator * divisor)
    except OverflowError:
        return math.inf if numerator > 0 else -math.inf


def reduce_sum(
    value: Tensor,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the sum of each group, accumulated stably."""
    data = value._data
    if axes == tuple(range(value.ndim)):
        if value.dtype.kind == "floating":
            total = _stable_float_sum([float(value) for value in data])
        else:
            total = builtins.sum(data)
        return PythonStorage.from_values([total], dtype)
    _, output_shape, groups = reduction_groups(
        value, axes, keepdims, scalar_as_vector=True
    )
    if value.dtype.kind == "floating":
        values = [
            _stable_float_sum([float(data[index]) for index in group])
            for group in groups
        ]
    else:
        values = [builtins.sum((data[index] for index in group)) for group in groups]
    return PythonStorage.from_values(values, dtype)
