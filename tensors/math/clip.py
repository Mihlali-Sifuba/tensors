"""Elementwise clipping to constant bounds."""

from __future__ import annotations

import math
from typing import Any

from ..dtype import result_dtype
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


class Clip:
    """Clip values using a zero subgradient at either finite boundary."""

    @staticmethod
    def forward(
        value: Tensor,
        *,
        min_value: int | float | None,
        max_value: int | float | None,
    ) -> Tensor:
        _validate_bounds(min_value, max_value)
        values = []
        for item in value._data:
            if min_value is not None and item < min_value:
                item = min_value
            if max_value is not None and item > max_value:
                item = max_value
            values.append(item)
        dtype = value.dtype
        if min_value is not None:
            dtype = result_dtype(dtype, min_value)
        if max_value is not None:
            dtype = result_dtype(dtype, max_value)
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

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> list[Tensor]:
        value = inputs[0]
        min_value = kwargs.get("min_value")
        max_value = kwargs.get("max_value")
        _validate_bounds(min_value, max_value)
        mask = Clip._mask(value, min_value, max_value)
        return [Tensor(
            [upstream * weight for upstream, weight in zip(grad._data, mask)],
            dtype=grad.dtype,
            shape=value.shape,
        )]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        from ..ops._utils import masked_value_graph, zero_like_graph

        value = inputs[0]
        min_value = kwargs.get("min_value")
        max_value = kwargs.get("max_value")
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


def clip(
    value: Any,
    min_value: int | float | None = None,
    max_value: int | float | None = None,
) -> Any:
    """Clip each value to the inclusive interval defined by the bounds."""
    from ..variable import Variable

    _validate_bounds(min_value, max_value)
    if isinstance(value, Variable):
        return Variable._from_operation(
            Clip.forward(
                value.data,
                min_value=min_value,
                max_value=max_value,
            ),
            "clip",
            Clip,
            [value],
            min_value=min_value,
            max_value=max_value,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Clip.forward(value, min_value=min_value, max_value=max_value)


__all__ = ["Clip", "clip"]
