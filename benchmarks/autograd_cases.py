"""Reverse-mode and higher-order automatic-differentiation benchmarks."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkCase


def _first_derivative_case(size: int) -> BenchmarkCase:
    value = ts.Variable(
        [1.0 + index / size for index in range(size)],
        name="x",
    )
    output = ts.sum(value ** 2.0 + value * 3.0)

    def run():
        return ts.grad(output, value)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == value.shape
        assert math.isclose(float(result[0]), 5.0)

    return BenchmarkCase(
        name=f"autograd.grad/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description="functional first derivative of an elementwise expression",
    )


def _backward_case(size: int) -> BenchmarkCase:
    value = ts.Variable(
        [1.0 + index / size for index in range(size)],
        name="x",
    )
    output = ts.sum(value ** 2.0 + value * 3.0)

    def run() -> None:
        ts.backward(output)

    def validate() -> None:
        run()
        assert isinstance(value.grad, ts.Tensor)
        assert value.grad.shape == value.shape
        assert math.isclose(float(value.grad[0]), 5.0)

    return BenchmarkCase(
        name=f"autograd.backward/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description="reverse pass that publishes gradients on Variables",
    )


def _derivative_graph_case(size: int) -> BenchmarkCase:
    value = ts.Variable(
        [1.0 + index / size for index in range(size)],
        name="x",
    )
    output = ts.sum(value ** 3.0)

    def run():
        return ts.grad(output, value, create_graph=True)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Variable)
        assert result.shape == value.shape
        assert result.requires_grad

    return BenchmarkCase(
        name=f"autograd.create_graph/{size}",
        run=run,
        validate=validate,
        work_items=size,
        gc_enabled=True,
        description="first derivative recorded as a differentiable graph",
    )


def _second_derivative_case() -> BenchmarkCase:
    def run():
        value = ts.Variable([2.0], name="x")
        output = value ** 3.0
        first = ts.grad(output, value, create_graph=True)
        return ts.grad(first, value)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert math.isclose(float(result.item()), 12.0)

    return BenchmarkCase(
        name="autograd.second_derivative/scalar",
        run=run,
        validate=validate,
        work_items=1,
        gc_enabled=True,
        description="fresh scalar expression and two reverse-mode passes",
    )


def _hessian_case(size: int) -> BenchmarkCase:
    value = ts.Variable(
        [1.0 + index / size for index in range(size)],
        name="x",
    )
    output = ts.sum(value ** 3.0)

    def run():
        return ts.hessian(output, value)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == value.shape + value.shape
        assert math.isclose(float(result[0, 0]), 6.0)
        assert result[0, 1] == 0.0

    return BenchmarkCase(
        name=f"autograd.hessian/{size}",
        run=run,
        validate=validate,
        work_items=size * size,
        gc_enabled=True,
        description="complete Hessian of a scalar separable polynomial",
    )


def cases() -> list[BenchmarkCase]:
    return [
        _first_derivative_case(1_024),
        _backward_case(1_024),
        _derivative_graph_case(256),
        _second_derivative_case(),
        _hessian_case(8),
    ]
