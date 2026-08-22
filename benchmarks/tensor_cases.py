"""Tensor-kernel and linear-algebra benchmark cases."""

from __future__ import annotations

import math
from collections.abc import Callable

import tensors as ts

from .runner import BenchmarkCase


def _elementwise_case(
    name: str,
    size: int,
    operation: Callable[[ts.Tensor, ts.Tensor], ts.Tensor],
    expected: float,
) -> BenchmarkCase:
    left = ts.full((size,), 1.5)
    right = ts.full((size,), 2.0)

    def run() -> ts.Tensor:
        return operation(left, right)

    def validate() -> None:
        result = run()
        assert result.shape == (size,)
        assert result[0] == expected
        assert result[-1] == expected

    return BenchmarkCase(
        name=name,
        run=run,
        validate=validate,
        work_items=size,
        description=f"elementwise operation over {size} values",
    )


def _matmul_case(size: int) -> BenchmarkCase:
    left = ts.full((size, size), 0.25)
    right = ts.full((size, size), 0.5)
    expected = size * 0.125

    def run() -> ts.Tensor:
        return ts.matmul(left, right)

    def validate() -> None:
        result = run()
        assert result.shape == (size, size)
        assert math.isclose(float(result[0, 0]), expected)
        assert math.isclose(float(result[-1, -1]), expected)

    return BenchmarkCase(
        name=f"linalg.matmul/{size}x{size}",
        run=run,
        validate=validate,
        work_items=size ** 3,
        description=f"square matrix multiplication at order {size}",
    )


def cases() -> list[BenchmarkCase]:
    """Return tensor cases with inputs materialized outside timed regions."""
    benchmarks = [
        _elementwise_case(
            f"tensor.add/{size}",
            size,
            ts.add,
            3.5,
        )
        for size in (100, 10_000, 100_000)
    ]
    benchmarks.append(
        _elementwise_case(
            "tensor.multiply/10000",
            10_000,
            ts.multiply,
            3.0,
        )
    )

    broadcast_left = ts.full((256, 1), 1.0)
    broadcast_right = ts.full((1, 256), 2.0)

    def broadcast_add() -> ts.Tensor:
        return broadcast_left + broadcast_right

    def validate_broadcast() -> None:
        result = broadcast_add()
        assert result.shape == (256, 256)
        assert result[0, 0] == 3.0
        assert result[-1, -1] == 3.0

    benchmarks.append(
        BenchmarkCase(
            name="tensor.broadcast_add/256x256",
            run=broadcast_add,
            validate=validate_broadcast,
            work_items=256 * 256,
            description="broadcast (256, 1) and (1, 256) before addition",
        )
    )

    reduction_size = 100_000
    reduction_input = ts.full((reduction_size,), 2.0)

    def reduce_sum() -> ts.Tensor:
        return ts.sum(reduction_input)

    def validate_sum() -> None:
        assert reduce_sum().item() == reduction_size * 2.0

    benchmarks.append(
        BenchmarkCase(
            name="reduction.sum/100000",
            run=reduce_sum,
            validate=validate_sum,
            work_items=reduction_size,
            description="full reduction with stable floating-point summation",
        )
    )

    def reduce_mean() -> ts.Tensor:
        return ts.mean(reduction_input)

    def validate_mean() -> None:
        assert reduce_mean().item() == 2.0

    benchmarks.append(
        BenchmarkCase(
            name="reduction.mean/100000",
            run=reduce_mean,
            validate=validate_mean,
            work_items=reduction_size,
            description="full floating-point mean reduction",
        )
    )

    variance_size = 10_000
    variance_input = ts.Tensor(
        [float(index % 17) for index in range(variance_size)],
    )

    def reduce_variance() -> ts.Tensor:
        return ts.variance(variance_input)

    def validate_variance() -> None:
        result = reduce_variance()
        assert result.size == 1
        assert math.isfinite(float(result.item()))
        assert result.item() > 0.0

    benchmarks.append(
        BenchmarkCase(
            name="reduction.variance/10000",
            run=reduce_variance,
            validate=validate_variance,
            work_items=variance_size,
            description="full variance reduction over non-constant values",
        )
    )

    benchmarks.extend(_matmul_case(size) for size in (16, 32, 64))

    norm_size = 10_000
    norm_input = ts.full((norm_size,), 3.0)

    def vector_norm() -> ts.Tensor:
        return ts.norm(norm_input)

    def validate_norm() -> None:
        expected = 3.0 * math.sqrt(norm_size)
        assert math.isclose(float(vector_norm().item()), expected)

    benchmarks.append(
        BenchmarkCase(
            name="linalg.norm/10000",
            run=vector_norm,
            validate=validate_norm,
            work_items=norm_size,
            description="Euclidean norm of a flat vector",
        )
    )
    return benchmarks
