"""Reference summation for the Python backend.

Integer groups are summed exactly. Floating groups use the package's
deterministic pairwise tree (`docs/summation-semantics.md`), the same one every
backend implements, so a floating sum is identical wherever it runs.
"""

from __future__ import annotations
from tensors.backend.python.storage import PythonStorage
from tensors.backend.python.kernels.reductions.pairwise import pairwise_float_sum
from tensors.backend.storage import Storage
from tensors.dtype import DataType

import builtins
from tensors.utils.reductions import reduction_groups


def reduce_sum(
    value_values,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Storage:
    """Return the sum of each group in logical order."""
    data = value_values
    if axes == tuple(range(len(input_shape))):
        if dtype.kind == "floating":
            total = pairwise_float_sum(data, dtype)
        else:
            total = builtins.sum(data)
        return PythonStorage.from_values([total], dtype)
    _, output_shape, groups = reduction_groups(
        input_shape, axes, keepdims, scalar_as_vector=True
    )
    if dtype.kind == "floating":
        values = [
            pairwise_float_sum((data[index] for index in group), dtype)
            for group in groups
        ]
    else:
        values = [builtins.sum((data[index] for index in group)) for group in groups]
    return PythonStorage.from_values(values, dtype)
