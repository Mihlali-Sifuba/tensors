"""Adam optimisation algorithm."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tensor import Tensor
from ..math import sqrt
from .optimizer import Optimizer

if TYPE_CHECKING:
    from ..variable import Variable


class Adam(Optimizer):
    """Adam optimiser with per-parameter adaptive learning rates.

    Implements the algorithm from *Adam: A Method for Stochastic
    Optimization* (Kingma & Ba, 2015).

    Hyper-parameters
    ----------------
    learning_rate : float
        Step size (default: 1e-3).
    betas : tuple[float, float]
        Coefficients for computing running averages of gradient and
        its square (default: (0.9, 0.999)).
    eps : float
        Term added to denominator to improve numerical stability
        (default: 1e-8).
    """

    def __init__(
        self,
        parameters: Iterable[Variable],
        learning_rate: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if len(betas) != 2:
            raise ValueError("betas must contain exactly two coefficients")
        if not all(math.isfinite(beta) and 0 <= beta < 1 for beta in betas):
            raise ValueError("betas must be finite and in the interval [0, 1)")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps must be positive and finite")

        super().__init__(parameters)
        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        self._state: dict[int, dict[str, Tensor | int]] = {}

    def step(self) -> None:
        """Apply one Adam update to every managed parameter."""
        b1 = self.beta1
        b2 = self.beta2
        lr = self.learning_rate
        eps = self.eps

        for param in self.parameters:
            grad = self._gradient_for(param)
            if grad is None:
                continue

            sid = id(param)
            state = self._state.get(sid)
            if state is not None:
                current_m = state["m"]
                current_v = state["v"]
                assert isinstance(current_m, Tensor)
                assert isinstance(current_v, Tensor)
                if (
                    current_m.shape != grad.shape
                    or current_m.dtype != grad.dtype
                    or current_v.shape != grad.shape
                    or current_v.dtype != grad.dtype
                ):
                    state = None

            if state is None:
                state = {
                    "step": 0,
                    "m": Tensor([0.0] * grad.size, dtype=grad.dtype, shape=grad.shape),
                    "v": Tensor([0.0] * grad.size, dtype=grad.dtype, shape=grad.shape),
                }
                self._state[sid] = state

            state["step"] = int(state["step"]) + 1
            step_count = int(state["step"])
            m = state["m"]
            v = state["v"]
            assert isinstance(m, Tensor)
            assert isinstance(v, Tensor)

            # Update biased moments
            m_new = b1 * m + (1.0 - b1) * grad
            v_new = b2 * v + (1.0 - b2) * (grad * grad)
            state["m"] = m_new
            state["v"] = v_new

            # Bias-corrected estimates
            m_hat = m_new / (1.0 - b1 ** step_count)
            v_hat = v_new / (1.0 - b2 ** step_count)

            step = lr * (m_hat / (sqrt(v_hat) + eps))
            param.data = param.data - step
