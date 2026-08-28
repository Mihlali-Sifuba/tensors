"""Graph tracing and replay benchmarks across topology and tensor width."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _elementwise_model(depth: int, topology: str) -> ts.Graph:
    if topology == "chain":
        @ts.Graph
        def model(value):
            result = value
            for _ in range(depth):
                result = result * 1.0001 + 0.1
            return result

        return model

    @ts.Graph
    def model(value):
        left = value
        right = value
        for _ in range(depth):
            left = left * 1.0001 + 0.1
            right = right / 1.0002 - 0.1
        return left + right

    return model


def _expected(depth: int, topology: str) -> float:
    if topology == "chain":
        result = 1.0
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result
    left = 1.0
    right = 1.0
    for _ in range(depth):
        left = left * 1.0001 + 0.1
        right = right / 1.0002 - 0.1
    return left + right


def _elementwise_cases(
    depth: int,
    width: int,
    topology: str,
    *,
    legacy_name: bool = False,
    backends: frozenset[BenchmarkBackend] | None = None,
) -> list[BenchmarkCase]:
    model = _elementwise_model(depth, topology)
    value = ts.full((width,), 1.0)
    expected = _expected(depth, topology)
    operation_factor = 2 if topology == "chain" else 5
    if legacy_name:
        trace_name = f"graph.trace/depth-{depth}"
        replay_name = f"graph.replay/depth-{depth}"
    else:
        prefix = f"{topology}/depth-{depth}/width-{width}"
        trace_name = f"graph.trace/{prefix}"
        replay_name = f"graph.replay/{prefix}"

    def trace():
        return model(value)

    def validate_trace() -> None:
        result = trace()
        assert result.shape == (width,)
        assert math.isclose(float(result.data[0]), expected)

    trace_case = BenchmarkCase(
        name=trace_name,
        run=trace,
        validate=validate_trace,
        work_items=operation_factor * depth * width,
        gc_enabled=True,
        description=f"fresh {topology} graph trace",
        layer="graph",
        backends=backends,
    )

    trace()
    computation = model.computation

    def replay() -> ts.Tensor:
        return computation.forward()

    def validate_replay() -> None:
        result = replay()
        assert result.shape == (width,)
        assert math.isclose(float(result[0]), expected)

    replay_case = BenchmarkCase(
        name=replay_name,
        run=replay,
        validate=validate_replay,
        work_items=operation_factor * depth * width,
        description=f"recorded {topology} graph replay",
        layer="graph",
        backends=backends,
    )
    return [trace_case, replay_case]


def _matrix_cases(
    depth: int,
    size: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> list[BenchmarkCase]:
    @ts.Graph
    def model(value, weight, bias):
        result = value
        for _ in range(depth):
            result = result @ weight + bias
        return result

    value = ts.full((size, size), 0.25)
    weight = ts.full((size, size), 1.0 / size)
    bias = ts.full((size,), 0.1)
    expected = 0.25 + depth * 0.1
    prefix = f"matrix/depth-{depth}/size-{size}"

    def trace():
        return model(value, weight, bias)

    def validate_trace() -> None:
        result = trace()
        assert result.shape == (size, size)
        assert math.isclose(float(result.data[0, 0]), expected)

    trace_case = BenchmarkCase(
        name=f"graph.trace/{prefix}",
        run=trace,
        validate=validate_trace,
        work_items=depth * size ** 3,
        gc_enabled=True,
        description="fresh matrix-heavy graph trace",
        layer="graph",
        backends=backends,
    )

    trace()
    computation = model.computation

    def replay() -> ts.Tensor:
        return computation.forward()

    def validate_replay() -> None:
        result = replay()
        assert result.shape == (size, size)
        assert math.isclose(float(result[0, 0]), expected)

    replay_case = BenchmarkCase(
        name=f"graph.replay/{prefix}",
        run=replay,
        validate=validate_replay,
        work_items=depth * size ** 3,
        description="recorded matrix-heavy graph replay",
        layer="graph",
        backends=backends,
    )
    return [trace_case, replay_case]


def core_cases() -> list[BenchmarkCase]:
    """Return the original compact scalar graph baselines."""
    return [
        case
        for depth in (10, 100)
        for case in _elementwise_cases(
            depth,
            1,
            "chain",
            legacy_name=True,
        )
    ]


def cases() -> list[BenchmarkCase]:
    """Return graph cases spanning chain, branch, and matrix topologies."""
    backend = ts.get_backend()
    benchmarks = core_cases()
    for topology in ("chain", "branch"):
        for depth in (10, 100):
            benchmarks.extend(_elementwise_cases(depth, 1_024, topology))
            if backend != "python":
                benchmarks.extend(_elementwise_cases(
                    depth,
                    100_000,
                    topology,
                    backends=_ACCELERATED,
                ))
    benchmarks.extend(_matrix_cases(4, 32, None))
    if backend != "python":
        benchmarks.extend(_matrix_cases(8, 128, _ACCELERATED))
    return benchmarks
