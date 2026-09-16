"""Shared structure for the index-of-extremum reductions.

``argmin`` and ``argmax`` reduce an axis to the position of its extreme value
by the same rule, differing only in which comparison wins. The axis handling
and the index semantics live here; each reduction owns its own module.
"""

from __future__ import annotations

from typing import Any

from tensors.backend import execute_argmax, execute_argmin
from tensors.dtype import int64
from tensors.graph.expression import as_tensor_operand
from tensors.tensor import Tensor
from tensors.utils.reductions import normalize_axes, reduction_shape


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
            "argmin": execute_argmin,
            "argmax": execute_argmax,
        }[
            operation
        ](value, axis, keepdims=keepdims, output_shape=output_shape)
        return Tensor._from_owned_storage(accelerated, dtype=int64, shape=output_shape)


def _arg_extremum(operation, value: Any, axis: int | None, keepdims: bool) -> Tensor:
    from tensors.variable import Variable

    tensor = value.data if isinstance(value, Variable) else value
    tensor = as_tensor_operand(tensor)
    return operation.forward(tensor, axis=axis, keepdims=keepdims)
