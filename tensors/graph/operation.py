"""The local mathematical contract implemented by graph operations.

An :class:`Operation` instance describes exactly one concrete invocation. It
owns the immutable configuration that defines its mathematical transformation
(a reduction axis, a cast dtype, a slice key) and nothing else. Runtime
operands are graph relationships rather than operation attributes, and values
produced by a particular forward pass belong to
:class:`~tensors.graph.computation.Computation`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from ..tensor import Tensor


class Operation(ABC):
    """One concrete mathematical invocation recorded in a computational graph.

    Subclasses implement ``forward`` and ``backward``. They may also implement
    ``backward_graph`` to support higher-order differentiation.

    An operation is immutable once constructed: its recorded mathematical
    meaning must not change while a graph still refers to it. Subclasses that
    take configuration declare it in ``__slots__`` and assign it with
    ``object.__setattr__`` inside ``__init__``.
    """

    __slots__ = ()

    #: Short label used for graph inspection and error messages.
    name: ClassVar[str] = "operation"

    def __setattr__(self, name: str, value: Any) -> None:
        """Reject mutation so a recorded invocation keeps its meaning."""
        raise AttributeError(
            f"{type(self).__name__} is immutable; construct a new operation "
            "instead of reconfiguring a recorded one"
        )

    def __delattr__(self, name: str) -> None:
        """Reject attribute removal for the same reason as assignment."""
        raise AttributeError(
            f"{type(self).__name__} is immutable; construct a new operation "
            "instead of reconfiguring a recorded one"
        )

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        """Calculate this invocation's eager output from its input tensors."""

    @abstractmethod
    def backward(
        self,
        gradient: Tensor,
        *inputs: Tensor,
    ) -> Iterable[Tensor]:
        """Return one numerical VJP contribution per input tensor."""

    def backward_graph(
        self,
        gradient: Any,
        *inputs: Any,
    ) -> Iterable[Any]:
        """Build a differentiable VJP and return one result per input.

        Operations that support higher-order differentiation override this.
        """
        raise NotImplementedError(
            f"Higher-order derivatives are not implemented for {self.name}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


__all__ = ["Operation"]
