"""Cross-cutting system and lifecycle benchmark cases."""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor

import tensors as ts

from .runner import BenchmarkCase


def _graph_build_case(depth: int, width: int) -> BenchmarkCase:
    """Measure node and edge bookkeeping while building a deep eager graph."""
    def run() -> ts.Variable:
        value = ts.Variable(ts.full((width,), 1.0), name="build_root")
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result

    def validate() -> None:
        result = run()
        assert result.size == width

    return BenchmarkCase(
        name=f"system.graph_build/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        gc_enabled=True,
        description="fresh deep graph with node, edge, and state capture",
        layer="graph",
    )


class _ThreadMLP(ts.Graph):
    """Minimal two-layer MLP used to exercise thread-local graph isolation."""

    def __init__(self) -> None:
        super().__init__()
        self.input_to_hidden = ts.Variable(
            ts.full((4, 8), 0.05),
            name="input_to_hidden",
        )
        self.hidden_to_output = ts.Variable(
            ts.full((8, 2), 0.05),
            name="hidden_to_output",
        )

    def forward(self, inputs):
        hidden = ts.relu(inputs @ self.input_to_hidden)
        return hidden @ self.hidden_to_output


def _thread_step(
    model: _ThreadMLP,
    optimizer: ts.optim.Optimizer,
    inputs: ts.Tensor,
    targets: ts.Tensor,
) -> float:
    prediction = model(inputs)
    loss = ts.mean((prediction - targets) ** 2.0)
    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()
    return float(loss.data.item())


def _threading_case(batch_size: int) -> BenchmarkCase:
    """Measure concurrent MLP steps that rely on thread-local graph state."""
    inputs = ts.full((batch_size, 4), 0.25)
    targets = ts.full((batch_size, 2), 0.1)
    def state():
        models = [_ThreadMLP() for _ in range(2)]
        optimizers = [
            ts.optim.SGD(model.parameters(), learning_rate=1e-4)
            for model in models
        ]
        return models, optimizers

    models, optimizers = state()
    pool = ThreadPoolExecutor(max_workers=2)

    def reset() -> None:
        nonlocal models, optimizers
        models, optimizers = state()

    def run() -> list[float]:
        futures = [
            pool.submit(
                _thread_step,
                models[index],
                optimizers[index],
                inputs,
                targets,
            )
            for index in range(2)
        ]
        return [future.result() for future in futures]

    def validate() -> None:
        results = run()
        assert len(results) == 2
        assert all(math.isfinite(value) for value in results)

    def teardown() -> None:
        pool.shutdown(wait=True)

    return BenchmarkCase(
        name=f"threading.mlp_isolated/{batch_size}",
        run=run,
        validate=validate,
        work_items=2 * batch_size,
        gc_enabled=True,
        description="two concurrent MLP steps in separate thread-local graphs",
        layer="training",
        reset=reset,
        teardown=teardown,
    )


def _equality_case(size: int) -> BenchmarkCase:
    left = ts.full((size,), 1.25)
    right = ts.full((size,), 1.25)

    def run() -> bool:
        return left == right

    def validate() -> None:
        assert run() is True

    return BenchmarkCase(
        name=f"system.equality/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description="full-buffer equality comparison between identical tensors",
    )


def _scalar_item_case() -> BenchmarkCase:
    value = ts.full((1,), 3.5)

    def run() -> float:
        return value.item()

    def validate() -> None:
        assert run() == 3.5

    return BenchmarkCase(
        name="system.scalar_item/1",
        run=run,
        validate=validate,
        work_items=1,
        description="single-element tensor extraction used by validation paths",
    )


def cases() -> list[BenchmarkCase]:
    """Return graph-scale, threading, and small public-API overhead cases."""
    return [
        _graph_build_case(1_000, 16),
        _threading_case(16),
        _equality_case(10_000),
        _scalar_item_case(),
    ]
