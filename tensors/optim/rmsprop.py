"""RMSprop optimisation algorithm."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tensor import Tensor
from ..math import sqrt
from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..variable import Variable


class RMSprop(Optimizer):
    """RMSprop optimiser with per-parameter adaptive learning rates.

    Hyper-parameters
    ----------------
    learning_rate : float
        Step size (default: 1e-2).
    rho : float
        Decay rate for the running average of squared gradients
        (default: 0.99).
    eps : float
        Term added to denominator to improve numerical stability
        (default: 1e-8).
    """

    def __init__(
        self,
        parameters: Iterable[Variable],
        learning_rate: float = 1e-2,
        rho: float = 0.99,
        eps: float = 1e-8,
    ) -> None:
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= rho < 1:
            raise ValueError("rho must be in the interval [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")

        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        self._state: dict[int, Tensor] = {}

    def step(self) -> None:
        """Apply one RMSprop update to every managed parameter."""
        for param in self.parameters:
            grad = self._gradient_for(param)
            if grad is None:
                continue

            sid = id(param)
            if sid not in self._state:
                self._state[sid] = Tensor(
                    [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape,
                )

            v = self._state[sid]

            # v = rho * v + (1 - rho) * grad^2
            v_new = self.rho * v + (1.0 - self.rho) * (grad * grad)
            self._state[sid] = v_new

            denom = sqrt(v_new) + self.eps
            param.data = param.data - self.learning_rate * (grad / denom)


__all__ = ["RMSprop"]
