"""RMSprop optimisation algorithm."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tensor import Tensor
from .optimizer import Optimizer
from .adam import _scaled_second_moment, _visible_second_moment

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
        self._scaled_state: dict[int, tuple[Tensor, Tensor]] = {}

    @property
    def rho(self) -> float:
        return self._rho

    @rho.setter
    def rho(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("rho must be numeric, not bool")
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("rho must be finite and in the interval [0, 1)")
        self._rho = value

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("eps must be numeric, not bool")
        if not math.isfinite(value) or value <= 0:
            raise ValueError("eps must be positive and finite")
        self._eps = value

    def step(self) -> None:
        """Apply one RMSprop update to every managed parameter."""
        pending = []
        for param, grad in self._prepared_gradients():
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
                scales = Tensor(
                    [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape
                )
                scaled_values = Tensor(
                    [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape
                )
            else:
                scaled_state = self._scaled_state.get(sid)
                if (
                    scaled_state is None
                    or scaled_state[0].shape != grad.shape
                    or scaled_state[0].dtype != grad.dtype
                ):
                    scale_data = [math.sqrt(float(value)) for value in v._data]
                    scales = Tensor(
                        scale_data, dtype=grad.dtype, shape=grad.shape
                    )
                    scaled_values = Tensor(
                        [1.0 if value else 0.0 for value in scale_data],
                        dtype=grad.dtype,
                        shape=grad.shape,
                    )
                else:
                    scales, scaled_values = scaled_state

            new_scales = []
            new_scaled_values = []
            visible_values = []
            parameter_values = []
            for parameter_value, gradient_value, scale, scaled in zip(
                param.data._data,
                grad._data,
                scales._data,
                scaled_values._data,
            ):
                gradient_value = float(gradient_value)
                new_scale, new_scaled = _scaled_second_moment(
                    float(scale), float(scaled), gradient_value, self.rho
                )
                root_moment = new_scale * math.sqrt(new_scaled)
                update = self.learning_rate * (
                    gradient_value / (root_moment + self.eps)
                )
                new_scales.append(new_scale)
                new_scaled_values.append(new_scaled)
                visible_values.append(
                    _visible_second_moment(new_scale, new_scaled)
                )
                parameter_values.append(float(parameter_value) - update)

            visible_state = Tensor(
                visible_values, dtype=grad.dtype, shape=grad.shape
            )
            scaled_state = (
                Tensor(new_scales, dtype=grad.dtype, shape=grad.shape),
                Tensor(new_scaled_values, dtype=grad.dtype, shape=grad.shape),
            )
            new_parameter = Tensor(
                parameter_values, dtype=param.dtype, shape=param.shape
            )
            pending.append((param, sid, new_parameter, visible_state, scaled_state))

        for param, sid, new_parameter, visible_state, scaled_state in pending:
            self._state[sid] = visible_state
            self._scaled_state[sid] = scaled_state
            param.data = new_parameter


__all__ = ["RMSprop"]
