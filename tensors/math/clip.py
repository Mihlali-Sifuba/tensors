"""Elementwise clipping to constant bounds."""

from __future__ import annotations

import math
from typing import Any, overload

from .._typing import TensorData, TensorLike, TensorResult, TensorValue
from ..backend import execute_clip, execute_clip_gradient
from ..dtype import result_dtype
from ..ops.operation import Operation
from ..tensor import Tensor


def _validate_bound(name: str, value: int | float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number or None")
    if isinstance(value, float) and math.isnan(value):
        raise ValueError(f"{name} cannot be NaN")


def _validate_bounds(
    min_value: int | float | None,
    max_value: int | float | None,
) -> None:
    _validate_bound("min_value", min_value)
    _validate_bound("max_value", max_value)
    if min_value is None and max_value is None:
        raise ValueError("clip requires min_value, max_value, or both")
    if (
        min_value is not None
        and max_value is not None
        and min_value > max_value
    ):
        raise ValueError("min_value cannot be greater than max_value")


class Clip(Operation):
    """Clip values using a zero subgradient at either finite boundary."""

    __slots__ = ("min_value", "max_value")
    name = "clip"

    def __init__(
        self,
        *,
        min_value: int | float | None,
        max_value: int | float | None,
    ) -> None:
        object.__setattr__(self, "min_value", min_value)
        object.__setattr__(self, "max_value", max_value)

    def forward(self, value: Tensor) -> Tensor:
        min_value = self.min_value
        max_value = self.max_value
        _validate_bounds(min_value, max_value)
        dtype = value.dtype
        if min_value is not None:
            dtype = result_dtype(dtype, min_value)
        if max_value is not None:
            dtype = result_dtype(dtype, max_value)
        accelerated = execute_clip(
            value,
            min_value,
            max_value,
            dtype=dtype,
        )
        if accelerated is not None:
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=value.shape)
        values = []
        for item in value._data:
            if min_value is not None and item < min_value:
                item = min_value
            if max_value is not None and item > max_value:
                item = max_value
            values.append(item)
        return Tensor(values, dtype=dtype, shape=value.shape)

    @staticmethod
    def _mask(
        value: Tensor,
        min_value: int | float | None,
        max_value: int | float | None,
    ) -> list[float]:
        mask = []
        for item in value._data:
            if isinstance(item, float) and math.isnan(item):
                mask.append(math.nan)
                continue
            above_minimum = min_value is None or item > min_value
            below_maximum = max_value is None or item < max_value
            mask.append(1.0 if above_minimum and below_maximum else 0.0)
        return mask

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> list[Tensor]:
        value = inputs[0]
        min_value = self.min_value
        max_value = self.max_value
        _validate_bounds(min_value, max_value)
        accelerated = execute_clip_gradient(
            grad,
            value,
            min_value,
            max_value,
        )
        if accelerated is not None:
            return [Tensor._from_owned_storage(
                accelerated,
                dtype=grad.dtype,
                shape=value.shape,
            )]
        mask = Clip._mask(value, min_value, max_value)
        return [Tensor(
            [upstream * weight for upstream, weight in zip(grad._data, mask)],
            dtype=grad.dtype,
            shape=value.shape,
        )]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        from ..ops._utils import masked_value_graph, zero_like_graph

        value = inputs[0]
        min_value = self.min_value
        max_value = self.max_value
        _validate_bounds(min_value, max_value)
        if any(
            isinstance(item, float) and math.isnan(item)
            for item in value.data._data
        ):
            raise ValueError(
                "Higher-order derivatives of clip are undefined at NaN"
            )
        mask = Tensor(
            Clip._mask(value.data, min_value, max_value),
            dtype=grad.dtype,
            shape=value.shape,
        )
        return [masked_value_graph(grad, mask) + zero_like_graph(value)]


@overload
def clip(
    value: TensorValue,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> TensorValue: ...


@overload
def clip(
    value: TensorData,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> Tensor: ...


def clip(
    value: TensorLike,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> TensorResult:
    """Clip each value to the inclusive interval defined by the bounds."""
    from ..variable import Variable

    _validate_bounds(min_value, max_value)
    if isinstance(value, Variable):
        operation = Clip(min_value=min_value, max_value=max_value)
        return Variable._from_operation(
            operation.forward(value.data),
            operation,
            (value,),
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Clip(min_value=min_value, max_value=max_value).forward(value)


__all__ = ["Clip", "clip"]
