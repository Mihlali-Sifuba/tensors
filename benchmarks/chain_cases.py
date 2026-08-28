"""Expression-chain benchmarks for launch overhead and future fusion work."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _chain_case(
    depth: int,
    width: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    value = ts.full((width,), 1.0)
    expected = 1.0
    for _ in range(depth):
        expected = expected * 1.0001 + 0.1

    def run() -> ts.Tensor:
        result = value
        for _ in range(depth):
            result = result * 1.0001 + 0.1
        return result

    def validate() -> None:
        result = run()
        assert result.shape == (width,)
        assert math.isclose(float(result[0]), expected)
        assert math.isclose(float(result[-1]), expected)

    return BenchmarkCase(
        name=f"chain.elementwise/depth-{depth}/width-{width}",
        run=run,
        validate=validate,
        work_items=2 * depth * width,
        description="unfused multiply-add expression chain",
        backends=backends,
    )


def cases() -> list[BenchmarkCase]:
    """Vary both chain depth and tensor width to expose launch amortization."""
    backend = ts.get_backend()
    combinations: list[tuple[int, int, frozenset[BenchmarkBackend] | None]] = []
    for depth in (1, 10, 100):
        for width in (1, 1_000, 100_000):
            if backend == "python" and depth == 100 and width == 100_000:
                continue
            combinations.append((depth, width, None))
    if backend != "python":
        combinations.extend(
            (depth, 1_000_000, _ACCELERATED)
            for depth in (1, 10, 100)
        )
    return [
        _chain_case(depth, width, backends)
        for depth, width, backends in combinations
    ]
