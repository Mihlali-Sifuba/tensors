"""Base contract for parameter optimisation algorithms."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tensor import Tensor

if TYPE_CHECKING:
    from ..variable import Variable


class Optimizer(ABC):
    """Base class for algorithms that update trainable Variables.

    Subclasses implement :meth:`step`; the common parameter ownership and
    gradient-reset behavior live here.
    """

    def __init__(self, parameters: Iterable[Variable]) -> None:
        from ..variable import Variable

        unique_parameters: list[Variable] = []
        seen: set[int] = set()
        for index, parameter in enumerate(parameters):
            if not isinstance(parameter, Variable):
                raise TypeError(
                    f"optimizer parameter {index} must be a Variable, got "
                    f"{type(parameter).__name__}"
                )
            identity = id(parameter)
            if identity not in seen:
                seen.add(identity)
                unique_parameters.append(parameter)
        self.parameters = tuple(unique_parameters)

    @property
    def learning_rate(self) -> float:
        """Return the current validated step size."""
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self._learning_rate = value

    def zero_grad(self) -> None:
        """Clear gradients from every parameter managed by this optimizer."""
        for parameter in self.parameters:
            parameter.grad = None

    @staticmethod
    def _gradient_for(parameter: Variable) -> Tensor | None:
        """Return a parameter-shaped gradient in the parameter's dtype."""
        gradient = parameter.grad
        if gradient is None:
            return None
        if not isinstance(gradient, Tensor):
            raise TypeError("Optimizer gradients must be Tensors")
        if gradient.shape != parameter.shape:
            raise ValueError(
                f"Gradient shape {gradient.shape} does not match parameter "
                f"shape {parameter.shape}"
            )
        if gradient.dtype != parameter.dtype:
            return gradient.astype(parameter.dtype)
        return gradient

    @abstractmethod
    def step(self) -> None:
        """Apply one parameter update using the current gradients."""


__all__ = ["Optimizer"]
