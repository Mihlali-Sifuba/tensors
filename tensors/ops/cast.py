"""Differentiable tensor dtype conversion."""

from __future__ import annotations

from typing import Any, List

from ..dtype import DataType
from ..tensor import Tensor


class Cast:
    """Convert a tensor dtype while preserving its differentiation path."""

    @staticmethod
    def forward(value: Tensor, *, dtype: DataType) -> Tensor:
        return value.astype(dtype)

    @staticmethod
    def backward(
        grad: Tensor,
        *inputs: Tensor,
        **kwargs: object,
    ) -> List[Tensor]:
        return [grad.astype(inputs[0].dtype)]

    @staticmethod
    def backward_graph(grad: Any, *inputs: Any, **kwargs: object) -> List[Any]:
        """Build a differentiable VJP converted to the input dtype."""
        return [grad.astype(inputs[0].dtype)]


__all__ = ["Cast"]
