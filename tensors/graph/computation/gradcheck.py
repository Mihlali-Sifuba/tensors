"""Finite-difference verification for reverse-mode gradient rules."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

from ..._typing import TensorResult
from ...tensor import Tensor
from ...utils.coordinates import linear_index_to_coordinates
from ...variable import Variable
from .autograd import grad
from ..state import isolated_graph_state, reset_graph_state


class GradcheckError(AssertionError):
    """Raised when an analytical gradient disagrees with finite differences."""


def _input_tensors(inputs: Any) -> tuple[Tensor, ...]:
    requested = (inputs,) if isinstance(inputs, (Tensor, Variable)) else tuple(inputs)
    if not requested:
        raise ValueError("gradcheck requires at least one input")

    tensors = []
    for index, value in enumerate(requested):
        tensor = value.data if isinstance(value, Variable) else value
        if not isinstance(tensor, Tensor):
            raise TypeError(f"gradcheck input {index} must be a Tensor or Variable")
        if tensor.dtype.typecode not in {"f", "d"}:
            raise TypeError(f"gradcheck input {index} must have a floating-point dtype")
        if any(not math.isfinite(float(item)) for item in tensor._data):
            raise ValueError(f"gradcheck input {index} must contain only finite values")
        tensors.append(tensor.clone())
    return tuple(tensors)


def _scalar_output(
    function: Callable[..., TensorResult],
    inputs: tuple[Tensor, ...],
) -> float:
    variables = tuple(Variable(value, requires_grad=False) for value in inputs)
    output = function(*variables)
    tensor = output.data if isinstance(output, Variable) else output
    if not isinstance(tensor, Tensor):
        raise TypeError("gradcheck function must return a Tensor or Variable")
    return math.fsum(float(value) for value in tensor._data)


def gradcheck(
    function: Callable[..., TensorResult],
    inputs: Tensor | Variable | Sequence[Tensor | Variable],
    *,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-3,
    raise_exception: bool = True,
) -> bool:
    """Compare reverse-mode gradients with central finite differences.

    The checked scalar objective is the sum of every element returned by
    ``function``. Float64 inputs are recommended for the default tolerances.
    """
    if not callable(function):
        raise TypeError("gradcheck function must be callable")
    if not isinstance(raise_exception, bool):
        raise TypeError("raise_exception must be a bool")
    if any(isinstance(value, bool) for value in (eps, atol, rtol)):
        raise TypeError("gradcheck tolerances must be numeric, not bools")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be positive and finite")
    if (
        not math.isfinite(atol)
        or not math.isfinite(rtol)
        or atol < 0
        or rtol < 0
    ):
        raise ValueError("tolerances must be non-negative and finite")

    originals = _input_tensors(inputs)
    with isolated_graph_state():
        analytical_inputs = tuple(
            Variable(value.clone(), requires_grad=True) for value in originals
        )
        output = function(*analytical_inputs)
        if not isinstance(output, Variable):
            raise TypeError(
                "gradcheck function must return a Variable connected to its inputs"
            )
        from ...math import sum

        objective = sum(output)
        analytical = grad(objective, analytical_inputs)
        analytical_values = [
            [0.0] * value.size if gradient is None else gradient.tolist()
            for value, gradient in zip(originals, analytical)
        ]

        for input_index, original in enumerate(originals):
            for element_index in range(original.size):
                positive = [value.clone() for value in originals]
                negative = [value.clone() for value in originals]
                positive_value = positive[input_index]._data[element_index]
                negative_value = negative[input_index]._data[element_index]
                coordinates = linear_index_to_coordinates(
                    element_index,
                    original.shape,
                )
                positive[input_index][coordinates] = positive_value + eps
                negative[input_index][coordinates] = negative_value - eps

                reset_graph_state()
                positive_output = _scalar_output(function, tuple(positive))
                reset_graph_state()
                negative_output = _scalar_output(function, tuple(negative))
                numerical = (positive_output - negative_output) / (2.0 * eps)
                actual = analytical_values[input_index][element_index]

                if not math.isclose(actual, numerical, abs_tol=atol, rel_tol=rtol):
                    message = (
                        f"Gradient mismatch at input {input_index}, element "
                        f"{element_index}: analytical={actual}, numerical={numerical}, "
                        f"atol={atol}, rtol={rtol}"
                    )
                    if raise_exception:
                        raise GradcheckError(message)
                    return False
    return True


__all__ = ["GradcheckError", "gradcheck"]
