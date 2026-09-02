"""Optimizer first-step and steady-state scaling benchmarks."""

from __future__ import annotations

import math
from collections.abc import Callable

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _optimizer_cases(
    name: str,
    optimizer_type: Callable[..., ts.optim.Optimizer],
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> list[BenchmarkCase]:
    def first_step() -> ts.Tensor:
        parameter = ts.Variable(ts.full((size,), 1.0))
        parameter.grad = ts.full((size,), 0.5)
        optimizer = optimizer_type([parameter], learning_rate=1e-3)
        optimizer.step()
        return parameter.data

    def validate_first_step() -> None:
        result = first_step()
        assert result.shape == (size,)
        assert math.isfinite(float(result[0]))

    first = BenchmarkCase(
        name=f"optimizer.{name}_first/{size}",
        run=first_step,
        validate=validate_first_step,
        work_items=size,
        description=f"{name} construction, state initialization, and first update",
        layer="optimizer",
        backends=backends,
    )

    def steady_state():
        parameter = ts.Variable(ts.full((size,), 1.0))
        parameter.grad = ts.full((size,), 0.5)
        optimizer = optimizer_type([parameter], learning_rate=1e-3)
        optimizer.step()
        return parameter, optimizer

    parameter = None
    optimizer = None

    def reset_steady() -> None:
        nonlocal parameter, optimizer
        parameter, optimizer = steady_state()

    def steady_step() -> ts.Tensor:
        if optimizer is None or parameter is None:
            reset_steady()
        assert optimizer is not None
        assert parameter is not None
        optimizer.step()
        return parameter.data

    def validate_steady_step() -> None:
        result = steady_step()
        assert result.shape == (size,)
        assert math.isfinite(float(result[0]))

    steady = BenchmarkCase(
        name=f"optimizer.{name}_steady/{size}",
        run=steady_step,
        validate=validate_steady_step,
        work_items=size,
        description=f"{name} update with initialized reusable state",
        layer="optimizer",
        backends=backends,
        reset=reset_steady,
    )
    return [first, steady]


def _many_parameter_case(
    name: str,
    optimizer_type: Callable[..., ts.optim.Optimizer],
    parameter_count: int,
    width: int,
) -> BenchmarkCase:
    """Measure grouped updates where launch overhead dominates each tensor."""
    def steady_state():
        parameters = [
            ts.Variable(ts.full((width,), 1.0 + index * 0.01))
            for index in range(parameter_count)
        ]
        for index, parameter in enumerate(parameters):
            parameter.grad = ts.full((width,), 0.1 + index * 0.001)
        optimizer = optimizer_type(parameters, learning_rate=1e-3)
        optimizer.step()
        return parameters, optimizer

    parameters = None
    optimizer = None

    def reset() -> None:
        nonlocal parameters, optimizer
        parameters, optimizer = steady_state()

    def run() -> ts.Tensor:
        if optimizer is None or parameters is None:
            reset()
        assert optimizer is not None
        assert parameters is not None
        optimizer.step()
        return parameters[-1].data

    def validate() -> None:
        result = run()
        assert result.shape == (width,)
        assert math.isfinite(float(result[0]))

    return BenchmarkCase(
        name=(
            f"optimizer.{name}_many/"
            f"parameters-{parameter_count}/width-{width}"
        ),
        run=run,
        validate=validate,
        work_items=parameter_count * width,
        description=(
            f"{name} steady update across many small parameter tensors"
        ),
        layer="optimizer",
        reset=reset,
    )


def cases() -> list[BenchmarkCase]:
    """Scale optimizer initialization and steady updates independently."""
    backend = ts.get_backend()
    sizes = [1_000, 10_000, 100_000]
    if backend != "python":
        sizes.append(1_000_000)
    optimizers = (
        ("sgd", ts.optim.SGD),
        ("adam", ts.optim.Adam),
        ("rmsprop", ts.optim.RMSprop),
    )
    benchmarks: list[BenchmarkCase] = []
    for size in sizes:
        restriction = _ACCELERATED if size > 100_000 else None
        for name, optimizer_type in optimizers:
            benchmarks.extend(_optimizer_cases(
                name,
                optimizer_type,
                size,
                restriction,
            ))
    for name, optimizer_type in optimizers:
        benchmarks.append(
            _many_parameter_case(name, optimizer_type, 64, 256)
        )
    return benchmarks
