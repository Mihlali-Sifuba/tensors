"""Axis traversal and softmax components shared by the Python nn kernels.

These are the pieces more than one reference kernel in the softmax family
needs. They work on plain Python floats so that no kernel has to apply a public
operation in order to evaluate its own reference result.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from tensors.utils.normalization import shifted_normalization

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _axis_layout(tensor: Tensor, axis: int) -> tuple[int, int, int]:
    """Return the row-major group sizes needed to traverse ``axis``."""
    before = 1
    for dimension in tensor.shape[:axis]:
        before *= dimension
    trailing = 1
    for dimension in tensor.shape[axis + 1 :]:
        trailing *= dimension
    return (before, tensor.shape[axis], trailing)


def _axis_positions(tensor: Tensor, axis: int) -> list[list[int]]:
    """Return the flat positions of every one-dimensional group along ``axis``."""
    before, axis_size, trailing = _axis_layout(tensor, axis)
    groups = []
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            groups.append(
                [group_start + offset + index * trailing for index in range(axis_size)]
            )
    return groups


def _floating_dtype(value: Tensor) -> DataType:
    """Return the dtype the softmax family produces for ``value``."""
    from tensors.dtype import float64

    return value.dtype if value.dtype.kind == "floating" else float64


def _softmax_values(value: Tensor, axis: int) -> list[float]:
    """Return softmax probabilities in row-major order."""
    from tensors.backend.python.kernels.nn.softmax import softmax

    storage = softmax(value, axis, dtype=_floating_dtype(value))
    return [float(item) for item in storage.buffer]


def _log_softmax_values(value: Tensor, axis: int) -> list[float]:
    """Return log-softmax values in row-major order."""
    from tensors.backend.python.kernels.nn.log_softmax import log_softmax

    storage = log_softmax(value, axis, dtype=_floating_dtype(value))
    return [float(item) for item in storage.buffer]


def _normalization_components(
    value: Tensor,
    axis: int,
) -> tuple[list[float], list[float]]:
    """Return softmax probabilities and accurately represented complements.

    The complement of the largest probability loses its significant digits to
    cancellation when computed as ``1 - p``, so it is taken from the shifted
    normalizer instead wherever the group is finite.
    """
    probabilities = _softmax_values(value, axis)
    complements = [0.0] * value.size
    for positions in _axis_positions(value, axis):
        group_values = [float(value._data[position]) for position in positions]
        if all(math.isfinite(item) for item in group_values):
            _, _, _, group_complements = shifted_normalization(group_values)
        else:
            group_complements = [
                1.0 - probabilities[position] for position in positions
            ]
        for position, complement in zip(positions, group_complements):
            complements[position] = complement
    return (probabilities, complements)
