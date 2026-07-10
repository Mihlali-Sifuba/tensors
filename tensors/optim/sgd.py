"""Stochastic gradient descent."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..autograd.variable import Variable


class SGD:
    """Update a sequence of trainable Variables with gradient descent."""

    def __init__(self, parameters: Iterable["Variable"], learning_rate: float) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        self.parameters = tuple(parameters)
        self.learning_rate = learning_rate

    def zero_grad(self) -> None:
        """Clear accumulated gradients from every managed parameter."""
        for parameter in self.parameters:
            parameter.grad = None

    def step(self) -> None:
        """Apply one in-place parameter update using current gradients."""
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            parameter.data = parameter.data - self.learning_rate * parameter.grad
