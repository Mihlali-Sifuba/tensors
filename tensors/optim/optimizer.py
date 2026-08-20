"""Base contract for parameter optimisation algorithms."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..variable import Variable


class Optimizer(ABC):
    """Base class for algorithms that update trainable Variables.

    Subclasses implement :meth:`step`; the common parameter ownership and
    gradient-reset behavior live here.
    """

    def __init__(self, parameters: Iterable[Variable]) -> None:
        unique_parameters: list[Variable] = []
        seen: set[int] = set()
        for parameter in parameters:
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                unique_parameters.append(parameter)
        self.parameters = tuple(unique_parameters)

    def zero_grad(self) -> None:
        """Clear gradients from every parameter managed by this optimizer."""
        for parameter in self.parameters:
            parameter.grad = None

    @abstractmethod
    def step(self) -> None:
        """Apply one parameter update using the current gradients."""


__all__ = ["Optimizer"]
