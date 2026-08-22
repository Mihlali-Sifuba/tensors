"""Stochastic gradient descent."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..backend import execute_sgd_update
from ..tensor import Tensor
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
        pending = []
        for parameter, gradient in self._prepared_gradients():
            storage = execute_sgd_update(
                parameter.data,
                gradient,
                self.learning_rate,
            )
            if storage is None:
                value = parameter.data - self.learning_rate * gradient
            else:
                value = Tensor(
                    storage,
                    dtype=parameter.dtype,
                    shape=parameter.shape,
                )
            pending.append((parameter, value))
        for parameter, value in pending:
            parameter.data = value
