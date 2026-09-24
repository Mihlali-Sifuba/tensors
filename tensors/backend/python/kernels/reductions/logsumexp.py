"""Reference the stable log-sum-exp for the Python backend."""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.storage import Storage
from tensors.dtype import DataType
import math
from tensors.utils.reductions import reduction_groups
from tensors.utils.normalization import shifted_normalization


def _group_value(value_values, indices: list[int]) -> float:
    """Return ``log(sum(exp(a[indices])))`` without avoidable overflow."""
    if not indices:
        raise ValueError("logsumexp is not defined over an empty reduction")
    if any((math.isnan(float(value_values[index])) for index in indices)):
        return math.nan
    group = [float(value_values[index]) for index in indices]
    maximum = max(group)
    if maximum == math.inf:
        return math.inf
    if maximum == -math.inf:
        return -math.inf
    _, correction, _, _ = shifted_normalization(group)
    return maximum + correction


def logsumexp(
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage | None:
    """Return the log-sum-exp of each group, shifted for stability."""
    _, output_shape, groups = reduction_groups(
        input_shape, axes, keepdims, scalar_as_vector=True
    )
    values = [_group_value(value_values, group) for group in groups]
    return PythonStorage.from_values(values, dtype)
