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


def _replay_new_inputs_case(
    depth: int,
    width: int,
) -> BenchmarkCase:
    """Measure compiled replay after swapping a leaf's value each call."""
    @ts.Graph
    def model(value):
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result

    value = ts.Variable(ts.full((width,), 1.0), name="replay_input")
    model(value)
    computation = model.computation
    new_data = ts.full((width,), 2.0)
    expected = 2.0
    for _ in range(depth):
        expected = expected * 1.0001 + 0.1

    def run() -> ts.Tensor:
        value.data = new_data
        return computation.forward()

    def validate() -> None:
        result = run()
        assert result.shape == (width,)
        assert math.isclose(float(result[0]), expected)

    return BenchmarkCase(
        name=f"graph.replay/new_inputs/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        gc_enabled=True,
        description="compiled replay after replacing a leaf value",
        layer="graph",
    )


def _release_cleanup_case(
    depth: int,
    width: int,
) -> BenchmarkCase:
    """Measure the trace-and-release lifecycle of a fresh computation."""
    @ts.Graph
    def model(value):
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result

    value = ts.full((width,), 1.0)

    def run() -> None:
        model(value)
        model.computation.release()

    def validate() -> None:
        run()

    return BenchmarkCase(
        name=f"graph.release/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        gc_enabled=True,
        description="fresh graph trace followed by computation release",
        layer="graph",
    )


def _nested_trace_case(depth: int, width: int) -> BenchmarkCase:
    """Measure hierarchical tracing without repeated upstream planning."""
    @ts.Graph
    def block(value):
        return value * 1.0001 + 0.1

    @ts.Graph
    def model(value):
        result = value
        for _ in range(depth):
            result = block(result)
        return result

    value = ts.full((width,), 1.0)
    expected = _expected(depth, "chain")

    def run():
        return model(value)

    def validate() -> None:
        result = run()
        assert result.shape == (width,)
        assert math.isclose(float(result.data[0]), expected)

    return BenchmarkCase(
        name=f"graph.trace/nested/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        gc_enabled=True,
        description="fresh hierarchical graph trace",
        layer="graph",
    )


def _multi_output_trace_case(
    depth: int,
    width: int,
    output_count: int,
) -> BenchmarkCase:
    """Measure one shared plan built from several output roots."""
    @ts.Graph
    def model(value):
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return tuple(result + index for index in range(output_count))

    value = ts.full((width,), 1.0)
    expected = _expected(depth, "chain")

    def run():
        return model(value)

    def validate() -> None:
        results = run()
        assert len(results) == output_count
        assert math.isclose(float(results[-1].data[0]), expected + output_count - 1)

    return BenchmarkCase(
        name=(
            f"graph.trace/multi_output/depth-{depth}/width-{width}/"
            f"outputs-{output_count}"
        ),
        run=run,
        validate=validate,
        work_items=(2 * depth + output_count) * width,
        gc_enabled=True,
        description="fresh shared-trunk multi-output graph trace",
        layer="graph",
    )


def _compiled_call_case(depth: int, width: int) -> BenchmarkCase:
    """Measure guarded Graph invocation with Tensor input rebinding."""
    model = _elementwise_model(depth, "chain")
    value = ts.full((width,), 1.0)
    expected = _expected(depth, "chain")
    model.compile(value)

    def run():
        return model(value)

    def validate() -> None:
        result = run()
        assert result.shape == (width,)
        assert math.isclose(float(result.data[0]), expected)

    return BenchmarkCase(
        name=f"graph.compiled/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        description="guarded graph call with compiled replay",
        layer="graph",
    )


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
    benchmarks.append(_replay_new_inputs_case(10, 1_024))
    benchmarks.append(_release_cleanup_case(10, 1_024))
    benchmarks.append(_nested_trace_case(100, 1))
    benchmarks.append(_multi_output_trace_case(100, 1, 32))
    benchmarks.append(_compiled_call_case(100, 1))
    return benchmarks
