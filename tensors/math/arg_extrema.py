"""Indices of minimum and maximum values along an axis."""

from __future__ import annotations

import math
from typing import Any

from .._typing import TensorLike
from ..dtype import int64
from ..tensor import Tensor
from ..utils.shape import index_to_coordinates
from ._reduction import reduction_groups


def _axis(axis: int | None, ndim: int) -> int | None:
    if axis is None:
        return None
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("argmin and argmax axis must be an integer or None")
    original = axis
    if axis < 0:
        axis += ndim
    if not 0 <= axis < ndim:
        raise ValueError(f"Axis {original} out of bounds for {ndim}D tensor")
    return axis


class _ArgExtremum:
    select_maximum = False

    @classmethod
    def forward(
        cls,
        value: Tensor,
        axis: int | None = None,
        keepdims: bool = False,
    ) -> Tensor:
        axis = _axis(axis, value.ndim)
        _, output_shape, groups = reduction_groups(
            value,
            axis,
            keepdims,
            scalar_as_vector=True,
        )
        if any(not group for group in groups):
            name = "argmax" if cls.select_maximum else "argmin"
            raise ValueError(f"Cannot compute {name} of empty tensor")

        indices = []
        for group in groups:
            nan_indices = [
                index
                for index in group
                if isinstance(value._data[index], float)
                and math.isnan(value._data[index])
            ]
            if nan_indices:
                selected = nan_indices[0]
            else:
                selected = group[0]
                for candidate in group[1:]:
                    candidate_value = value._data[candidate]
                    selected_value = value._data[selected]
                    if (
                        candidate_value > selected_value
                        if cls.select_maximum
                        else candidate_value < selected_value
                    ):
                        selected = candidate
            indices.append(
                selected
                if axis is None
                else index_to_coordinates(selected, value.shape)[axis]
            )
        return Tensor(indices, dtype=int64, shape=output_shape)


class ArgMax(_ArgExtremum):
    """Indices of maximum values, selecting the first tie."""

    select_maximum = True


class ArgMin(_ArgExtremum):
    """Indices of minimum values, selecting the first tie."""


def _arg_extremum(
    operation,
    value: Any,
    axis: int | None,
    keepdims: bool,
) -> Tensor:
    from ..variable import Variable

    tensor = value.data if isinstance(value, Variable) else value
    if not isinstance(tensor, Tensor):
        tensor = Tensor(tensor)
    return operation.forward(tensor, axis=axis, keepdims=keepdims)


def argmax(
    value: TensorLike,
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return first-occurrence indices of maximum values."""
    return _arg_extremum(ArgMax, value, axis, keepdims)


def argmin(
    value: TensorLike,
    axis: int | None = None,
    keepdims: bool = False,
) -> Tensor:
    """Return first-occurrence indices of minimum values."""
    return _arg_extremum(ArgMin, value, axis, keepdims)


__all__ = ["ArgMax", "ArgMin", "argmax", "argmin"]
