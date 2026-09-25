"""Axis traversal and softmax components shared by Python nn kernels."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from tensors.utils.normalization import shifted_normalization

if TYPE_CHECKING:
    from tensors.dtype import DataType
    from tensors.tensor import Tensor


def _axis_layout(shape: tuple[int, ...], axis: int) -> tuple[int, int, int]:
    """Return the row-major group sizes needed to traverse ``axis``."""
    before = math.prod(shape[:axis])
    trailing = math.prod(shape[axis + 1 :])
    return (before, shape[axis], trailing)


def _axis_positions(shape: tuple[int, ...], axis: int) -> list[list[int]]:
    """Return flat positions for every one-dimensional group along ``axis``."""
    before, axis_size, trailing = _axis_layout(shape, axis)
    groups = []
    for group in range(before):
        group_start = group * axis_size * trailing
        for offset in range(trailing):
            groups.append(
                [group_start + offset + index * trailing for index in range(axis_size)]
            )
    return groups


def _floating_dtype(dtype: DataType) -> DataType:
    """Return the dtype produced by normalizing values of ``dtype``."""
    from tensors.dtype import float64

    return dtype if dtype.kind == "floating" else float64


def _softmax_buffer_values(
    value_values: Sequence[Any],
    input_shape: tuple[int, ...],
    axis: int,
    value_dtype: DataType,
) -> list[float]:
    """Return softmax probabilities in row-major order."""
    from tensors.backend.python.kernels.nn.softmax import softmax

    storage = softmax(
        value_values,
        input_shape,
        axis,
        dtype=_floating_dtype(value_dtype),
    )
    return [float(item) for item in storage.buffer]


def _normalization_components(
    value_values: Sequence[Any],
    input_shape: tuple[int, ...],
    axis: int,
    value_dtype: DataType,
) -> tuple[list[float], list[float]]:
    """Return probabilities and accurately represented complements."""
    probabilities = _softmax_buffer_values(value_values, input_shape, axis, value_dtype)
    complements = [0.0] * math.prod(input_shape)
    for positions in _axis_positions(input_shape, axis):
        group_values = [float(value_values[position]) for position in positions]
        if all(math.isfinite(item) for item in group_values):
            _, _, _, group_complements = shifted_normalization(group_values)
        else:
            group_complements = [
                1.0 - probabilities[position] for position in positions
            ]
        for position, complement in zip(positions, group_complements):
            complements[position] = complement
    return (probabilities, complements)


def _softmax_values(value: Tensor, axis: int) -> list[float]:
    """Retain the Tensor-facing helper used by unmigrated loss kernels."""
    return _softmax_buffer_values(value._data, value.shape, axis, value.dtype)


def _log_softmax_values(value: Tensor, axis: int) -> list[float]:
    """Retain the Tensor-facing helper used by unmigrated loss kernels."""
    from tensors.backend.python.kernels.nn.log_softmax import log_softmax

    storage = log_softmax(
        value._data,
        value.shape,
        axis,
        dtype=_floating_dtype(value.dtype),
    )
    return [float(item) for item in storage.buffer]
