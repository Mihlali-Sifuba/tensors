"""Indices of minimum and maximum values along an axis."""

from __future__ import annotations
from tensors.backend import dispatch as backend_dispatch
from typing import Any
from .._typing import TensorLike
from ..dtype import int64
from ..tensor import Tensor
from ..graph.expression import as_tensor_operand
from ._reduction import normalize_axes, reduction_shape


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
        cls, value: Tensor, axis: int | None = None, keepdims: bool = False
    ) -> Tensor:
        axis = _axis(axis, value.ndim)
        axes = normalize_axes(value.ndim, axis)
        output_shape = reduction_shape(value.shape, axes, keepdims)
        if axis is None and (not keepdims):
            output_shape = (1,)
        operation = "argmax" if cls.select_maximum else "argmin"
        accelerated = {
            "argmin": backend_dispatch.execute_argmin,
            "argmax": backend_dispatch.execute_argmax,
        }[operation](value, axis, keepdims=keepdims, output_shape=output_shape)
        return Tensor._from_owned_storage(accelerated, dtype=int64, shape=output_shape)


class ArgMax(_ArgExtremum):
    """Indices of maximum values, selecting the first tie."""

    select_maximum = True


class ArgMin(_ArgExtremum):
    """Indices of minimum values, selecting the first tie."""


def _arg_extremum(operation, value: Any, axis: int | None, keepdims: bool) -> Tensor:
    from ..variable import Variable

    tensor = value.data if isinstance(value, Variable) else value
    tensor = as_tensor_operand(tensor)
    return operation.forward(tensor, axis=axis, keepdims=keepdims)


def argmax(
    value: TensorLike, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Return first-occurrence indices of maximum values."""
    return _arg_extremum(ArgMax, value, axis, keepdims)


def argmin(
    value: TensorLike, axis: int | None = None, keepdims: bool = False
) -> Tensor:
    """Return first-occurrence indices of minimum values."""
    return _arg_extremum(ArgMin, value, axis, keepdims)


__all__ = ["ArgMax", "ArgMin", "argmax", "argmin"]
