"""Shared static types for the public tensor API."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

if TYPE_CHECKING:
    from .tensor import Tensor
    from .variable import Variable


Scalar: TypeAlias = int | float
TensorIndex: TypeAlias = int | slice | tuple[int | slice, ...]
RawTensorData: TypeAlias = Union[Scalar, list[Any], array]
TensorData: TypeAlias = Union[RawTensorData, "Tensor"]
VariableData: TypeAlias = Union[TensorData, "Variable"]
TensorLike: TypeAlias = Union[TensorData, "Variable"]
TensorOperand: TypeAlias = Union[Scalar, "Tensor", "Variable"]
TensorResult: TypeAlias = Union["Tensor", "Variable"]
TensorValue = TypeVar(
    "TensorValue",
    bound=Union["Tensor", "Variable"],
)


__all__ = [
    "RawTensorData",
    "Scalar",
    "TensorData",
    "TensorIndex",
    "TensorLike",
    "TensorOperand",
    "TensorResult",
    "TensorValue",
    "VariableData",
]
