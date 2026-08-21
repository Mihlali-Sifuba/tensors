"""Stochastic gradient descent."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..variable import Variable


class SGD(Optimizer):
    """Update a sequence of trainable Variables with gradient descent."""

    def __init__(self, parameters: Iterable["Variable"], learning_rate: float) -> None:
        self.learning_rate = learning_rate
        super().__init__(parameters)

    def step(self) -> None:
        """Apply one in-place parameter update using current gradients."""
        for parameter in self.parameters:
            gradient = self._gradient_for(parameter)
            if gradient is None:
                continue
            parameter.data = parameter.data - self.learning_rate * gradient
