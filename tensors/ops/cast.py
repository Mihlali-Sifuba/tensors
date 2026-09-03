"""Differentiable tensor dtype conversion."""

from __future__ import annotations

from typing import Any, List

from ..dtype import DataType
from ..graph.operation import Operation
from ..tensor import Tensor


class Cast(Operation):
    """Convert a tensor dtype while preserving its differentiation path."""

    __slots__ = ("dtype",)
    name = "astype"

    def __init__(
        self,
        *,
        dtype: DataType,
    ) -> None:
        object.__setattr__(self, "dtype", dtype)

    def forward(self, value: Tensor) -> Tensor:
        dtype = self.dtype
        return value.astype(dtype)

    def backward(self, grad: Tensor, *inputs: Tensor) -> List[Tensor]:
        return [grad.astype(inputs[0].dtype)]

    def backward_graph(self, grad: Any, *inputs: Any) -> List[Any]:
        """Build a differentiable VJP converted to the input dtype."""
        return [grad.astype(inputs[0].dtype)]


__all__ = ["Cast"]
