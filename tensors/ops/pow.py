"""Element-wise exponentiation and its differentiation rules."""

from __future__ import annotations

import math
from typing import Any, List, Union

from ..dtype import float64, result_dtype
from ..tensor import Tensor, _broadcast_tensors
from ._utils import unbroadcast


Scalar = Union[int, float]


def _power_dtype(base: Tensor, exponent: Tensor | Scalar):
    """Choose a dtype that can represent the requested power operation."""
    if isinstance(exponent, Tensor):
        if base.dtype.typecode in {"f", "d"} or exponent.dtype.typecode in {"f", "d"}:
            return result_dtype(base.dtype, exponent)
        if any(value < 0 for value in exponent._data):
            return float64
        return result_dtype(base.dtype, exponent)

    if base.dtype.typecode in {"f", "d"}:
        return result_dtype(base.dtype, exponent)
    if isinstance(exponent, float) or exponent < 0:
        return float64
    return base.dtype


def _power(base: int | float, exponent: int | float) -> int | float:
    """Calculate a real-valued power with a clear domain error."""
    if isinstance(base, int) and isinstance(exponent, int) and exponent >= 0:
        return base ** exponent
    try:
        value = math.pow(base, exponent)
    except ValueError as exc:
        raise ValueError("power is not defined for these real-valued inputs") from exc
    except OverflowError as exc:
        raise OverflowError("power result is too large to represent") from exc
    return value


class Pow:
    """Element-wise exponentiation with reverse-mode gradient rules."""

    @staticmethod
    def forward(base: Tensor, exponent: Tensor | Scalar) -> Tensor:
        """Raise every element in ``base`` to ``exponent``."""
        dtype = _power_dtype(base, exponent)
        if isinstance(exponent, (int, float)):
            values = [_power(value, exponent) for value in base._data]
            return Tensor(values, dtype=dtype, shape=base.shape)
        if isinstance(exponent, Tensor):
            expanded_base, expanded_exponent = _broadcast_tensors(base, exponent)
            values = [
                _power(value, power)
                for value, power in zip(expanded_base._data, expanded_exponent._data)
            ]
            return Tensor(values, dtype=dtype, shape=expanded_base.shape)
        raise TypeError(f"Unsupported exponent type: {type(exponent)}")

    @staticmethod
    def forward_reverse(exponent: Tensor, base: Scalar) -> Tensor:
        """Return ``base`` raised element-wise to ``exponent``."""
        base_tensor = Tensor([base] * exponent.size, dtype=float64, shape=exponent.shape)
        return Pow.forward(base_tensor, exponent)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Return gradients for a power operation's differentiable inputs."""
        if len(inputs) == 2:
            base, exponent = inputs
            expanded_base, expanded_exponent = _broadcast_tensors(base, exponent)
            if any(value <= 0 for value in expanded_base._data):
                raise ValueError(
                    "power gradients with respect to a tensor exponent require positive bases"
                )
            output = Pow.forward(expanded_base, expanded_exponent)
            base_grad = Tensor(
                [
                    upstream * power * (value ** (power - 1))
                    for upstream, value, power in zip(
                        grad._data,
                        expanded_base._data,
                        expanded_exponent._data,
                    )
                ],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            exponent_grad = Tensor(
                [
                    upstream * value * math.log(base_value)
                    for upstream, value, base_value in zip(
                        grad._data,
                        output._data,
                        expanded_base._data,
                    )
                ],
                dtype=grad.dtype,
                shape=grad.shape,
            )
            return [
                unbroadcast(base_grad, base.shape),
                unbroadcast(exponent_grad, exponent.shape),
            ]

        exponent = kwargs.get("scalar")
        if not isinstance(exponent, (int, float)):
            raise TypeError("power scalar exponent must be an int or float")

        value = inputs[0]
        if kwargs.get("reverse", False):
            if exponent <= 0:
                raise ValueError(
                    "power gradients with respect to an exponent require a positive base"
                )
            output = Pow.forward_reverse(value, exponent)
            values = [
                upstream * result * math.log(exponent)
                for upstream, result in zip(grad._data, output._data)
            ]
            return [Tensor(values, dtype=grad.dtype, shape=value.shape)]

        values = [
            upstream * exponent * _power(base, exponent - 1)
            for upstream, base in zip(grad._data, value._data)
        ]
        return [Tensor(values, dtype=grad.dtype, shape=value.shape)]


def pow(base: Any, exponent: Any) -> Any:
    """Return the element-wise power of two Tensors, Variables, or scalars."""
    return base ** exponent


__all__ = ["Pow", "pow"]
