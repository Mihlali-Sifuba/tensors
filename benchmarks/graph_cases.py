"""Fresh Graph tracing and recorded Computation replay benchmarks."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkCase


def _model(depth: int) -> ts.Graph:
    @ts.Graph
    def model(value):
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result

    return model


def _trace_case(depth: int) -> BenchmarkCase:
    model = _model(depth)
    value = ts.Tensor([1.0])

    def run():
        return model(value)

    def validate() -> None:
        result = run()
        assert result.shape == (1,)
        assert math.isfinite(float(result.data.item()))
        assert len(model.nodes) == 1 + 2 * depth

    return BenchmarkCase(
        name=f"graph.trace/depth-{depth}",
        run=run,
        validate=validate,
        work_items=depth,
        gc_enabled=True,
        description="eager Graph call that records a fresh operation chain",
    )


def _replay_case(depth: int) -> BenchmarkCase:
    model = _model(depth)
    value = ts.Tensor([1.0])
    model(value)
    computation = model.computation

    def run() -> ts.Tensor:
        return computation.forward()

    def validate() -> None:
        result = run()
        assert result.shape == (1,)
        assert math.isfinite(float(result.item()))

    return BenchmarkCase(
        name=f"graph.replay/depth-{depth}",
        run=run,
        validate=validate,
        work_items=depth,
        description="recorded Computation.forward replay without retracing",
    )


def cases() -> list[BenchmarkCase]:
    return [
        _trace_case(10),
        _trace_case(100),
        _replay_case(10),
        _replay_case(100),
    ]
