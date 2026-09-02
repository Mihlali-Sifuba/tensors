"""Backend-native random-number-generation benchmark cases."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def cases() -> list[BenchmarkCase]:
    """Return seeded and per-family generation cases."""
    shape = (100_000,)
    benchmarks: list[BenchmarkCase] = []

    def seed_sequential() -> None:
        ts.random.seed(1234)

    def validate_seed_sequential() -> None:
        ts.random.seed(1234)
        first = ts.random.uniform(shape)
        ts.random.seed(1234)
        second = ts.random.uniform(shape)
        assert float(first[0]) == float(second[0])

    benchmarks.append(BenchmarkCase(
        name="random.seed/sequential",
        run=seed_sequential,
        validate=validate_seed_sequential,
        work_items=1,
        description="reset all backend RNG streams to a fixed seed",
        backends=_ACCELERATED,
    ))

    def uniform_draw() -> ts.Tensor:
        return ts.random.uniform(shape)

    def validate_uniform_draw() -> None:
        result = uniform_draw()
        assert result.shape == shape
        assert 0.0 <= float(result[0]) < 1.0

    benchmarks.append(BenchmarkCase(
        name="random.uniform/100000",
        run=uniform_draw,
        validate=validate_uniform_draw,
        work_items=100_000,
        description="backend-native uniform draws over 100000 values",
        backends=_ACCELERATED,
    ))

    def normal_draw() -> ts.Tensor:
        return ts.random.normal(shape)

    def validate_normal_draw() -> None:
        result = normal_draw()
        assert result.shape == shape
        assert math.isfinite(float(result[0]))

    benchmarks.append(BenchmarkCase(
        name="random.normal/100000",
        run=normal_draw,
        validate=validate_normal_draw,
        work_items=100_000,
        description="backend-native normal draws over 100000 values",
        backends=_ACCELERATED,
    ))

    def randint_draw() -> ts.Tensor:
        return ts.random.randint(shape, 0, 10)

    def validate_randint_draw() -> None:
        result = randint_draw()
        assert result.shape == shape
        assert 0 <= int(result[0]) < 10

    benchmarks.append(BenchmarkCase(
        name="random.randint/100000",
        run=randint_draw,
        validate=validate_randint_draw,
        work_items=100_000,
        description="backend-native integer draws over 100000 values",
        backends=_ACCELERATED,
    ))

    return benchmarks
