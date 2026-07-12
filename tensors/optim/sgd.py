"""Stochastic gradient descent."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..variable import Variable


class SGD(Optimizer):
    """Update a sequence of trainable Variables with gradient descent."""

    def __init__(self, parameters: Iterable["Variable"], learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        super().__init__(parameters)
        self.learning_rate = learning_rate

    def step(self) -> None:
        """Apply one in-place parameter update using current gradients."""
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data = parameter.data - self.learning_rate * parameter.grad
