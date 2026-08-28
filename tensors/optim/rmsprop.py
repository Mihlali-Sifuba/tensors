"""RMSprop optimisation algorithm."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..backend import execute_rmsprop_update, execute_rmsprop_updates
from ..creation import zeros
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

    def _batched_step(
        self,
        prepared: tuple[tuple[Variable, Tensor], ...],
    ) -> bool:
        """Apply one grouped native update for compatible parameter states."""
        if len(prepared) < 2:
            return False
        records = []
        for parameter, gradient in prepared:
            identity = id(parameter)
            visible = self._state.get(identity)
            if (
                visible is None
                or visible.shape != gradient.shape
                or visible.dtype != gradient.dtype
            ):
                zero_state = zeros(gradient.shape, dtype=gradient.dtype)
                scales = zero_state
                scaled_values = zero_state
            else:
                scaled_state = self._scaled_state.get(identity)
                if (
                    scaled_state is None
                    or scaled_state[0].shape != gradient.shape
                    or scaled_state[0].dtype != gradient.dtype
                ):
                    scale_data = [
                        math.sqrt(float(value)) for value in visible._data
                    ]
                    scales = Tensor(
                        scale_data,
                        dtype=gradient.dtype,
                        shape=gradient.shape,
                    )
                    scaled_values = Tensor(
                        [1.0 if value else 0.0 for value in scale_data],
                        dtype=gradient.dtype,
                        shape=gradient.shape,
                    )
                else:
                    scales, scaled_values = scaled_state
            records.append((
                parameter,
                gradient,
                identity,
                scales,
                scaled_values,
            ))

        accelerated = execute_rmsprop_updates(
            tuple(record[0].data for record in records),
            tuple(record[1] for record in records),
            tuple(record[3] for record in records),
            tuple(record[4] for record in records),
            rho=self.rho,
            learning_rate=self.learning_rate,
            epsilon=self.eps,
        )
        if accelerated is None:
            return False
        (
            parameter_storages,
            visible_storages,
            scale_storages,
            scaled_storages,
        ) = accelerated
        pending = []
        for record, parameter_storage, visible_storage, scale_storage, scaled_storage in zip(
            records,
            parameter_storages,
            visible_storages,
            scale_storages,
            scaled_storages,
        ):
            parameter, gradient, identity, _, _ = record
            visible_state = Tensor(
                visible_storage,
                dtype=gradient.dtype,
                shape=gradient.shape,
            )
            scaled_state = (
                Tensor(
                    scale_storage,
                    dtype=gradient.dtype,
                    shape=gradient.shape,
                ),
                Tensor(
                    scaled_storage,
                    dtype=gradient.dtype,
                    shape=gradient.shape,
                ),
            )
            value = Tensor(
                parameter_storage,
                dtype=parameter.dtype,
                shape=parameter.shape,
            )
            pending.append((
                parameter,
                identity,
                value,
                visible_state,
                scaled_state,
            ))
        for parameter, identity, value, visible, scaled in pending:
            self._state[identity] = visible
            self._scaled_state[identity] = scaled
            parameter.data = value
        return True

    def step(self) -> None:
        """Apply one RMSprop update to every managed parameter."""
        prepared = self._prepared_gradients()
        if self._batched_step(prepared):
            return

        pending = []
        for param, grad in prepared:
            sid = id(param)
            v = self._state.get(sid)
            if (
                v is None
                or v.shape != grad.shape
                or v.dtype != grad.dtype
            ):
                zero_state = zeros(grad.shape, dtype=grad.dtype)
                v = zero_state
                scales = zero_state
                scaled_values = zero_state
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

            accelerated = execute_rmsprop_update(
                param.data,
                grad,
                scales,
                scaled_values,
                rho=self.rho,
                learning_rate=self.learning_rate,
                epsilon=self.eps,
            )
            if accelerated is not None:
                (
                    parameter_storage,
                    visible_storage,
                    scale_storage,
                    scaled_storage,
                ) = accelerated
                visible_state = Tensor(
                    visible_storage,
                    dtype=grad.dtype,
                    shape=grad.shape,
                )
                scaled_state = (
                    Tensor(
                        scale_storage,
                        dtype=grad.dtype,
                        shape=grad.shape,
                    ),
                    Tensor(
                        scaled_storage,
                        dtype=grad.dtype,
                        shape=grad.shape,
                    ),
                )
                new_parameter = Tensor(
                    parameter_storage,
                    dtype=param.dtype,
                    shape=param.shape,
                )
                pending.append(
                    (param, sid, new_parameter, visible_state, scaled_state)
                )
                continue

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
