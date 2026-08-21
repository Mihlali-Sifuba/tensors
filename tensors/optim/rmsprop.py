"""RMSprop optimisation algorithm."""

from __future__ import annotations

import math
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
        self.learning_rate = learning_rate
        self.rho = rho
        self.eps = eps
        super().__init__(parameters)
        self._state: dict[int, Tensor] = {}

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("rho must be finite and in the interval [0, 1)")
        self._rho = value

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        if not math.isfinite(value) or value <= 0:
            raise ValueError("eps must be positive and finite")
        self._eps = value

    def step(self) -> None:
        """Apply one RMSprop update to every managed parameter."""
        for param in self.parameters:
            grad = self._gradient_for(param)
            if grad is None:
                continue

            sid = id(param)
            v = self._state.get(sid)
            if (
                v is None
                or v.shape != grad.shape
                or v.dtype != grad.dtype
            ):
                v = Tensor(
                    [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape,
                )
                self._state[sid] = v

            # v = rho * v + (1 - rho) * grad^2
            v_new = self.rho * v + (1.0 - self.rho) * (grad * grad)
            self._state[sid] = v_new

            denom = sqrt(v_new) + self.eps
            param.data = param.data - self.learning_rate * (grad / denom)


__all__ = ["RMSprop"]
