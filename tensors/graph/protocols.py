"""Structural interfaces implemented by computational graph operations."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, runtime_checkable

from ..tensor import Tensor


@runtime_checkable
class Operation(Protocol):
    """The forward and reverse-mode interface required by an operation node.

    Operation classes satisfy this protocol structurally: they only need to
    provide compatible ``forward`` and ``backward`` callables. Inheriting from
    ``Operation`` is not required.
    """

    @staticmethod
    def forward(*inputs: Any, **kwargs: Any) -> Tensor:
        """Calculate an eager output from input tensors."""
        ...

    @staticmethod
    def backward(
        gradient: Tensor,
        *inputs: Tensor,
        **kwargs: Any,
    ) -> Iterable[Tensor]:
        """Return one numerical VJP contribution per input."""
        ...


@runtime_checkable
class HigherOrderOperation(Operation, Protocol):
    """An operation whose VJP can itself be recorded and differentiated."""

    @staticmethod
    def backward_graph(
        gradient: Any,
        *inputs: Any,
        **kwargs: Any,
    ) -> Iterable[Any]:
        """Build a differentiable VJP and return one result per input."""
        ...


@runtime_checkable
class ReverseOperation(Operation, Protocol):
    """An operation supporting a scalar on the left-hand side."""

    @staticmethod
    def forward_reverse(input: Tensor, scalar: int | float) -> Tensor:
        """Calculate ``scalar op input`` for a non-commutative operation."""
        ...


__all__ = ["HigherOrderOperation", "Operation", "ReverseOperation"]
