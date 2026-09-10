"""Shared static types for the public tensor API."""

from __future__ import annotations

from array import array
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar, Union

if TYPE_CHECKING:
    from .graph.node import VariableNode
    from .backend.storage import Storage
    from .tensor import Tensor
    from .variable import Variable


Scalar: TypeAlias = int | float
TensorIndex: TypeAlias = int | slice | tuple[int | slice, ...]
RawTensorData: TypeAlias = Union[Scalar, list[Any], array]
TensorData: TypeAlias = Union[RawTensorData, "Storage", "Tensor"]
VariableData: TypeAlias = Union[TensorData, "Variable"]
TensorLike: TypeAlias = Union[TensorData, "Variable"]
TensorOperand: TypeAlias = Union[Scalar, "Tensor", "Variable"]
TensorResult: TypeAlias = Union["Tensor", "Variable"]
#: A value an expression over the graph can be written in terms of: a runtime
#: Variable, or the vertex naming a value the graph has not calculated.
GraphOperand: TypeAlias = Union["Variable", "VariableNode"]
TensorValue = TypeVar(
    "TensorValue",
    bound=Union["Tensor", "Variable"],
)


__all__ = [
    "GraphOperand",
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
