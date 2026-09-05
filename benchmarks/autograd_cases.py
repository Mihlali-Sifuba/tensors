"""Reverse-mode and higher-order automatic-differentiation benchmarks."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


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
        layer="autograd",
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

    def reset() -> None:
        value.grad = None

    return BenchmarkCase(
        name=f"autograd.backward/{size}",
        run=run,
        validate=validate,
        work_items=size,
        layer="autograd",
        description="reverse pass that publishes gradients on Variables",
        reset=reset,
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
        layer="autograd",
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
        layer="autograd",
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
        layer="autograd",
        description="complete Hessian of a scalar separable polynomial",
    )


def _forward_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    value = ts.Variable(ts.linspace(1.0, 2.0, size), name="forward_x")

    def run():
        return ts.sum(value ** 2.0 + value * 3.0)

    def validate() -> None:
        result = run()
        assert result.size == 1
        assert math.isfinite(float(result.data.item()))

    return BenchmarkCase(
        name=f"autograd.forward/{size}",
        run=run,
        validate=validate,
        work_items=size,
        gc_enabled=True,
        description="fresh elementwise differentiable forward expression",
        layer="autograd",
        backends=backends,
    )


def _forward_backward_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    base = ts.linspace(1.0, 2.0, size)

    def run():
        value = ts.Variable(base, name="step_x")
        output = ts.sum(value ** 2.0 + value * 3.0)
        ts.backward(output)
        return value.grad

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == (size,)
        assert math.isclose(float(result[0]), 5.0)

    return BenchmarkCase(
        name=f"autograd.forward_backward/{size}",
        run=run,
        validate=validate,
        work_items=size,
        gc_enabled=True,
        description="fresh forward graph followed by reverse mode",
        layer="autograd",
        backends=backends,
    )


def _accumulation_case(size: int) -> BenchmarkCase:
    value = ts.Variable(ts.linspace(1.0, 2.0, size), name="accumulate_x")
    output = ts.sum(value ** 2.0)
    value.grad = ts.zeros((size,))

    def run() -> None:
        ts.backward(output)

    def validate() -> None:
        run()
        assert isinstance(value.grad, ts.Tensor)
        assert value.grad.shape == (size,)

    def reset() -> None:
        value.grad = ts.zeros((size,))

    return BenchmarkCase(
        name=f"autograd.accumulate/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description="reverse pass accumulated into an existing gradient",
        layer="autograd",
        reset=reset,
    )


def _matrix_backward_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    left = ts.Variable(ts.full((size, size), 0.25), name="matrix_left")
    right = ts.Variable(ts.full((size, size), 0.5), name="matrix_right")
    output = ts.sum(left @ right)

    def run() -> None:
        left.grad = None
        right.grad = None
        ts.backward(output)

    def validate() -> None:
        run()
        assert isinstance(left.grad, ts.Tensor)
        assert isinstance(right.grad, ts.Tensor)
        assert left.grad.shape == left.shape
        assert right.grad.shape == right.shape

    return BenchmarkCase(
        name=f"autograd.matrix_backward/{size}x{size}",
        run=run,
        validate=validate,
        work_items=2 * size ** 3,
        description="reverse pass through square matrix multiplication",
        layer="autograd",
        backends=backends,
    )


def _planned_chain_backward_case(
    depth: int,
    width: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    """Measure repeated reverse execution through a precompiled scalar chain."""
    value = ts.Variable(ts.full((width,), 1.0), name="planned_chain")
    output = value
    for _ in range(depth):
        output = output * 1.0001 + 0.1
    computation = ts.graph.Computation(output)
    seed = ts.full((width,), 1.0)
    expected = 1.0001 ** depth

    def run():
        computation.backward(seed)
        return value.grad

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == (width,)
        assert math.isclose(float(result[0]), expected)

    def reset() -> None:
        value.grad = None

    return BenchmarkCase(
        name=f"autograd.planned_chain/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=depth * width,
        description="reused compiled reverse plan for a scalar elementwise chain",
        layer="autograd",
        backends=backends,
        reset=reset,
    )


def core_cases() -> list[BenchmarkCase]:
    """Return the compact historical automatic-differentiation baselines."""
    return [
        _first_derivative_case(1_024),
        _backward_case(1_024),
        _derivative_graph_case(256),
        _second_derivative_case(),
        _hessian_case(8),
    ]


def _jacobian_case(size: int) -> BenchmarkCase:
    value = ts.Variable(ts.linspace(1.0, 2.0, size), name="jacobian_x")
    output = value ** 2.0

    def run():
        return ts.jacobian(output, value)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == (size, size)
        assert math.isclose(float(result[0, 0]), 2.0)
        assert result[0, 1] == 0.0

    return BenchmarkCase(
        name=f"autograd.jacobian/{size}",
        run=run,
        validate=validate,
        work_items=size * size,
        gc_enabled=True,
        layer="autograd",
        description="complete Jacobian of an elementwise vector expression",
    )


def _third_derivative_case() -> BenchmarkCase:
    def run():
        value = ts.Variable([2.0], name="third_x")
        output = value ** 3.0
        first = ts.grad(output, value, create_graph=True)
        second = ts.grad(first, value, create_graph=True)
        return ts.grad(second, value)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert math.isclose(float(result.item()), 6.0)

    return BenchmarkCase(
        name="autograd.third_derivative/scalar",
        run=run,
        validate=validate,
        work_items=1,
        gc_enabled=True,
        layer="autograd",
        description="fresh scalar expression and three reverse-mode passes",
    )


def _gradcheck_case(size: int) -> BenchmarkCase:
    value = ts.Variable(ts.linspace(1.0, 2.0, size), name="gradcheck_x")

    def objective(v):
        return ts.sum(v ** 2.0 + v * 3.0)

    def run() -> bool:
        return ts.gradcheck(objective, value)

    def validate() -> None:
        assert run() is True

    return BenchmarkCase(
        name=f"autograd.gradcheck/{size}",
        run=run,
        validate=validate,
        work_items=size,
        gc_enabled=True,
        layer="autograd",
        description="reverse-mode gradients verified by finite differences",
    )


def _broadcast_backward_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    left = ts.Variable(ts.full((size, 1), 1.0), name="broadcast_left")
    right = ts.Variable(ts.full((1, size), 2.0), name="broadcast_right")
    output = left + right

    def run() -> None:
        left.grad = None
        right.grad = None
        ts.backward(output)

    def validate() -> None:
        run()
        assert isinstance(left.grad, ts.Tensor)
        assert isinstance(right.grad, ts.Tensor)
        assert left.grad.shape == left.shape
        assert right.grad.shape == right.shape
        assert math.isclose(float(left.grad[0, 0]), size)
        assert math.isclose(float(right.grad[0, 0]), size)

    return BenchmarkCase(
        name=f"autograd.broadcast_backward/{size}x{size}",
        run=run,
        validate=validate,
        work_items=2 * size * size,
        gc_enabled=True,
        description="reverse pass that reduces a broadcasted addition gradient",
        layer="autograd",
        backends=backends,
    )


def _demand_matmul_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> list[BenchmarkCase]:
    """Compare reverse passes that request one, both, or all gradients.

    A matrix product calculates a separate VJP per operand, so requesting
    only one of them should cost noticeably less than requesting both.
    """
    left = ts.Variable(ts.full((size, size), 0.5), name="demand_left")
    right = ts.Variable(ts.full((size, size), 0.25), name="demand_right")
    output = ts.sum(left @ right)
    seed = ts.ones((1,))

    def single():
        return ts.grad(output, left, seed)

    def both():
        return ts.grad(output, (left, right), seed)

    def publish() -> None:
        ts.backward(output, seed)

    def reset() -> None:
        left.grad = None
        right.grad = None

    def validate_single() -> None:
        result = single()
        assert isinstance(result, ts.Tensor)
        assert result.shape == left.shape

    def validate_both() -> None:
        results = both()
        assert len(results) == 2
        assert all(isinstance(result, ts.Tensor) for result in results)

    def validate_publish() -> None:
        publish()
        assert isinstance(left.grad, ts.Tensor)
        assert isinstance(right.grad, ts.Tensor)

    shared = {
        "work_items": size * size,
        "layer": "autograd",
        "backends": backends,
    }
    return [
        BenchmarkCase(
            name=f"autograd.demand/grad_one/{size}x{size}",
            run=single,
            validate=validate_single,
            description="reverse pass requesting one matrix-product operand",
            **shared,
        ),
        BenchmarkCase(
            name=f"autograd.demand/grad_both/{size}x{size}",
            run=both,
            validate=validate_both,
            description="reverse pass requesting both operands",
            **shared,
        ),
        BenchmarkCase(
            name=f"autograd.demand/backward/{size}x{size}",
            run=publish,
            validate=validate_publish,
            reset=reset,
            description="reverse pass publishing every reachable gradient",
            **shared,
        ),
    ]


def _demand_frozen_case(
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    """Measure a reverse pass whose second operand is never requested."""
    value = ts.Variable(ts.full((size, size), 0.5), name="demand_value")
    frozen = ts.Variable(
        ts.full((size, size), 0.25),
        name="demand_frozen",
        requires_grad=False,
    )
    output = ts.sum(value @ frozen)
    seed = ts.ones((1,))

    def run():
        return ts.grad(output, value, seed)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Tensor)
        assert result.shape == value.shape

    return BenchmarkCase(
        name=f"autograd.demand/frozen_operand/{size}x{size}",
        run=run,
        validate=validate,
        work_items=size * size,
        layer="autograd",
        description="reverse pass skipping an unrequested operand VJP",
        backends=backends,
    )


def _demand_create_graph_case(size: int) -> BenchmarkCase:
    """Measure a differentiable reverse pass under the same demand model."""
    left = ts.Variable(
        [1.0 + index / size for index in range(size)],
        name="demand_graph_left",
    )
    right = ts.Variable(
        [0.5 + index / size for index in range(size)],
        name="demand_graph_right",
    )
    output = ts.sum(left * right + left ** 2.0)

    def run():
        return ts.grad(output, left, create_graph=True)

    def validate() -> None:
        result = run()
        assert isinstance(result, ts.Variable)
        assert result.shape == left.shape

    return BenchmarkCase(
        name=f"autograd.demand/create_graph/{size}",
        run=run,
        validate=validate,
        work_items=size,
        layer="autograd",
        description="differentiable reverse pass requesting one input",
    )


def cases() -> list[BenchmarkCase]:
    """Return phase, topology, accumulation, and higher-order benchmarks."""
    backend = ts.get_backend()
    benchmarks = core_cases()
    benchmarks.extend((
        _forward_case(1_024, None),
        _forward_backward_case(1_024, None),
        _accumulation_case(1_024),
        _matrix_backward_case(16, None),
        _planned_chain_backward_case(10, 1_024, None),
        _planned_chain_backward_case(100, 1_024, None),
        _jacobian_case(8),
        _third_derivative_case(),
        _gradcheck_case(8),
        _broadcast_backward_case(64, None),
        _demand_frozen_case(16, None),
        _demand_create_graph_case(256),
    ))
    benchmarks.extend(_demand_matmul_case(16, None))
    if backend != "python":
        benchmarks.extend((
            _forward_case(100_000, _ACCELERATED),
            _forward_backward_case(100_000, _ACCELERATED),
            _matrix_backward_case(64, _ACCELERATED),
            _planned_chain_backward_case(
                100,
                100_000,
                _ACCELERATED,
            ),
            _broadcast_backward_case(256, _ACCELERATED),
            _demand_frozen_case(64, _ACCELERATED),
        ))
        benchmarks.extend(_demand_matmul_case(64, _ACCELERATED))
    return benchmarks
