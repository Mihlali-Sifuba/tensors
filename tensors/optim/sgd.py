"""Stochastic gradient descent."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..backend import execute_sgd_update, execute_sgd_updates
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
        prepared = self._prepared_gradients()
        if len(prepared) > 1:
            parameters = tuple(parameter.data for parameter, _ in prepared)
            gradients = tuple(gradient for _, gradient in prepared)
            storages = execute_sgd_updates(
                parameters,
                gradients,
                self.learning_rate,
            )
            if storages is not None:
                pending = tuple(
                    (
                        parameter,
                        Tensor(
                            storage,
                            dtype=parameter.dtype,
                            shape=parameter.shape,
                        ),
                    )
                    for (parameter, _), storage in zip(prepared, storages)
                )
                for parameter, value in pending:
                    parameter.data = value
                return

        pending = []
        for parameter, gradient in prepared:
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
