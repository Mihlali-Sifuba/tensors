"""Adam optimisation algorithm."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING

from ..tensor import Tensor
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
        if len(betas) != 2:
            raise ValueError("betas must contain exactly two coefficients")

        self.learning_rate = learning_rate
        self.beta1, self.beta2 = betas
        self.eps = eps
        super().__init__(parameters)
        self._state: dict[int, dict[str, Tensor | int | float]] = {}

    @property
    def beta1(self) -> float:
        return self._beta1

    @beta1.setter
    def beta1(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("beta1 must be numeric, not bool")
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("beta1 must be finite and in the interval [0, 1)")
        self._beta1 = value

    @property
    def beta2(self) -> float:
        return self._beta2

    @beta2.setter
    def beta2(self, value: float) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("beta2 must be numeric, not bool")
        if not math.isfinite(value) or not 0 <= value < 1:
            raise ValueError("beta2 must be finite and in the interval [0, 1)")
        self._beta2 = value

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
        """Apply one Adam update to every managed parameter."""
        b1 = self.beta1
        b2 = self.beta2
        lr = self.learning_rate
        eps = self.eps

        pending = []
        for param, grad in self._prepared_gradients():
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
                current_state: dict[str, Tensor | int | float] = {
                    "step": 0,
                    "m": Tensor([0.0] * grad.size, dtype=grad.dtype, shape=grad.shape),
                    "v": Tensor([0.0] * grad.size, dtype=grad.dtype, shape=grad.shape),
                    "v_scale": Tensor(
                        [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape
                    ),
                    "v_scaled": Tensor(
                        [0.0] * grad.size, dtype=grad.dtype, shape=grad.shape
                    ),
                    "beta1_product": 1.0,
                    "beta2_product": 1.0,
                }
            else:
                current_state = state

            step_count = int(current_state["step"]) + 1
            m = current_state["m"]
            v = current_state["v"]
            assert isinstance(m, Tensor)
            assert isinstance(v, Tensor)
            scales = current_state.get("v_scale")
            scaled_values = current_state.get("v_scaled")
            if not isinstance(scales, Tensor) or not isinstance(scaled_values, Tensor):
                scale_data = [math.sqrt(float(value)) for value in v._data]
                scales = Tensor(scale_data, dtype=grad.dtype, shape=grad.shape)
                scaled_values = Tensor(
                    [1.0 if value else 0.0 for value in scale_data],
                    dtype=grad.dtype,
                    shape=grad.shape,
                )

            beta1_product = float(current_state.get("beta1_product", b1 ** (step_count - 1))) * b1
            beta2_product = float(current_state.get("beta2_product", b2 ** (step_count - 1))) * b2
            first_correction = 1.0 - beta1_product
            second_correction = 1.0 - beta2_product

            moment_values = []
            visible_second_values = []
            new_scales = []
            new_scaled_values = []
            parameter_values = []
            for parameter_value, gradient_value, moment, scale, scaled in zip(
                param.data._data,
                grad._data,
                m._data,
                scales._data,
                scaled_values._data,
            ):
                gradient_value = float(gradient_value)
                moment_value = _stable_weighted_sum(
                    b1, float(moment), 1.0 - b1, gradient_value
                )
                new_scale, new_scaled = _scaled_second_moment(
                    float(scale), float(scaled), gradient_value, b2
                )
                root_second_moment = new_scale * math.sqrt(new_scaled)
                ratio = _product_quotient(
                    [moment_value, math.sqrt(second_correction)],
                    [first_correction, root_second_moment + eps * math.sqrt(second_correction)],
                )
                update = lr * ratio

                moment_values.append(moment_value)
                new_scales.append(new_scale)
                new_scaled_values.append(new_scaled)
                visible_second_values.append(
                    _visible_second_moment(new_scale, new_scaled)
                )
                parameter_values.append(float(parameter_value) - update)

            new_state: dict[str, Tensor | int | float] = {
                "step": step_count,
                "m": Tensor(moment_values, dtype=grad.dtype, shape=grad.shape),
                "v": Tensor(
                    visible_second_values, dtype=grad.dtype, shape=grad.shape
                ),
                "v_scale": Tensor(new_scales, dtype=grad.dtype, shape=grad.shape),
                "v_scaled": Tensor(
                    new_scaled_values, dtype=grad.dtype, shape=grad.shape
                ),
                "beta1_product": beta1_product,
                "beta2_product": beta2_product,
            }
            new_parameter = Tensor(
                parameter_values,
                dtype=param.dtype,
                shape=param.shape,
            )
            pending.append((param, sid, new_parameter, new_state))

        for param, sid, new_parameter, new_state in pending:
            self._state[sid] = new_state
            param.data = new_parameter


def _stable_weighted_sum(
    left_weight: float,
    left: float,
    right_weight: float,
    right: float,
) -> float:
    """Return a two-term weighted sum with reliable cancellation."""
    from ..math.sum import _stable_float_sum

    return _stable_float_sum([
        left_weight * left,
        right_weight * right,
    ])


def _scaled_second_moment(
    scale: float,
    scaled: float,
    gradient: float,
    decay: float,
) -> tuple[float, float]:
    """Update a squared average without explicitly squaring a huge value."""
    new_scale = max(scale, abs(gradient))
    if new_scale == 0.0:
        return 0.0, 0.0
    previous_ratio = scale / new_scale
    gradient_ratio = abs(gradient) / new_scale
    new_scaled = (
        decay * scaled * previous_ratio * previous_ratio
        + (1.0 - decay) * gradient_ratio * gradient_ratio
    )
    return new_scale, new_scaled


def _visible_second_moment(scale: float, scaled: float) -> float:
    """Materialize a second moment for inspection when it fits in a float."""
    if scale == 0.0 or scaled == 0.0:
        return 0.0
    try:
        return _product_quotient([scale, scale, scaled], [1.0])
    except OverflowError:
        return math.inf


def _product_quotient(
    numerators: list[float],
    denominators: list[float],
) -> float:
    """Evaluate a finite product quotient as one exact binary ratio."""
    numerator = 1
    denominator = 1
    for value in numerators:
        value_numerator, value_denominator = float(value).as_integer_ratio()
        numerator *= value_numerator
        denominator *= value_denominator
    for value in denominators:
        value_numerator, value_denominator = float(value).as_integer_ratio()
        numerator *= value_denominator
        denominator *= value_numerator
    try:
        return numerator / denominator
    except OverflowError:
        return math.inf if numerator * denominator > 0 else -math.inf
