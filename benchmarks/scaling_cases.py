"""Public-API scaling curves for backend crossover analysis."""

from __future__ import annotations

import math
from collections.abc import Callable

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _binary_case(
    operation_name: str,
    size: int,
    left: ts.Tensor,
    right: ts.Tensor,
    operation: Callable[[ts.Tensor, ts.Tensor], ts.Tensor],
    expected: float,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    def run() -> ts.Tensor:
        return operation(left, right)

    def validate() -> None:
        result = run()
        assert result.shape == (size,)
        assert math.isclose(float(result[0]), expected)
        assert math.isclose(float(result[-1]), expected)

    return BenchmarkCase(
        name=f"scaling.{operation_name}/{size}",
        run=run,
        validate=validate,
        work_items=size,
        description=f"public elementwise {operation_name} over {size} values",
        backends=backends,
    )


def _matmul_case(
    rows: int,
    contraction: int,
    columns: int,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    left = ts.full((rows, contraction), 0.25)
    right = ts.full((contraction, columns), 0.5)
    expected = contraction * 0.125

    def run() -> ts.Tensor:
        return ts.matmul(left, right)

    def validate() -> None:
        result = run()
        assert result.shape == (rows, columns)
        assert math.isclose(float(result[0, 0]), expected)
        assert math.isclose(float(result[-1, -1]), expected)

    return BenchmarkCase(
        name=f"scaling.matmul/{rows}x{contraction}x{columns}",
        run=run,
        validate=validate,
        work_items=rows * contraction * columns,
        description="public rectangular matrix multiplication",
        backends=backends,
    )


def cases() -> list[BenchmarkCase]:
    """Return size curves, limiting prohibitive cases to accelerated backends."""
    backend = ts.get_backend()
    sizes = [100, 10_000, 100_000]
    if backend != "python":
        sizes.extend((1_000_000, 10_000_000))

    benchmarks: list[BenchmarkCase] = []
    for size in sizes:
        restriction = _ACCELERATED if size > 100_000 else None
        left = ts.full((size,), 1.5)
        right = ts.full((size,), 2.0)
        operations = (
            ("add", ts.add, 3.5),
            ("multiply", ts.multiply, 3.0),
            ("divide", ts.divide, 0.75),
            ("power", ts.pow, 2.25),
        )
        benchmarks.extend(
            _binary_case(
                name,
                size,
                left,
                right,
                operation,
                expected,
                restriction,
            )
            for name, operation, expected in operations
        )

        def run_exp(values=left) -> ts.Tensor:
            return ts.exp(values)

        def validate_exp(run=run_exp, expected_size=size) -> None:
            result = run()
            assert result.shape == (expected_size,)
            assert math.isclose(float(result[0]), math.exp(1.5))

        benchmarks.append(BenchmarkCase(
            name=f"scaling.exp/{size}",
            run=run_exp,
            validate=validate_exp,
            work_items=size,
            description="public elementwise exponential scaling curve",
            backends=restriction,
        ))

        varying = ts.linspace(0.0, 1.0, size)

        def reduce_sum(values=left) -> ts.Tensor:
            return ts.sum(values)

        def validate_sum(run=reduce_sum, expected_size=size) -> None:
            assert math.isclose(float(run().item()), expected_size * 1.5)

        benchmarks.append(BenchmarkCase(
            name=f"scaling.sum/{size}",
            run=reduce_sum,
            validate=validate_sum,
            work_items=size,
            description="public full sum reduction scaling curve",
            backends=restriction,
        ))

        def reduce_mean(values=left) -> ts.Tensor:
            return ts.mean(values)

        def validate_mean(run=reduce_mean) -> None:
            assert math.isclose(float(run().item()), 1.5)

        benchmarks.append(BenchmarkCase(
            name=f"scaling.mean/{size}",
            run=reduce_mean,
            validate=validate_mean,
            work_items=size,
            description="public full mean reduction scaling curve",
            backends=restriction,
        ))

        def reduce_variance(values=varying) -> ts.Tensor:
            return ts.variance(values)

        def validate_variance(run=reduce_variance) -> None:
            assert float(run().item()) > 0.0

        benchmarks.append(BenchmarkCase(
            name=f"scaling.variance/{size}",
            run=reduce_variance,
            validate=validate_variance,
            work_items=size,
            description="public variance reduction scaling curve",
            backends=restriction,
        ))

        def reduce_norm(values=left) -> ts.Tensor:
            return ts.norm(values)

        def validate_norm(run=reduce_norm, expected_size=size) -> None:
            assert math.isclose(
                float(run().item()),
                1.5 * math.sqrt(expected_size),
            )

        benchmarks.append(BenchmarkCase(
            name=f"scaling.norm/{size}",
            run=reduce_norm,
            validate=validate_norm,
            work_items=size,
            description="public Euclidean norm scaling curve",
            backends=restriction,
        ))

    broadcast_dimensions = [256]
    if backend != "python":
        broadcast_dimensions.extend((512, 1024, 2048))
    for dimension in broadcast_dimensions:
        restriction = _ACCELERATED if dimension > 256 else None
        left = ts.full((dimension, 1), 1.0)
        right = ts.full((1, dimension), 2.0)

        def broadcast_add(a=left, b=right) -> ts.Tensor:
            return a + b

        def validate_broadcast(run=broadcast_add, size=dimension) -> None:
            result = run()
            assert result.shape == (size, size)
            assert result[0, 0] == 3.0

        benchmarks.append(BenchmarkCase(
            name=f"scaling.broadcast_add/{dimension}x{dimension}",
            run=broadcast_add,
            validate=validate_broadcast,
            work_items=dimension * dimension,
            description="public two-axis broadcast addition scaling curve",
            backends=restriction,
        ))

    axis_shapes = [(128, 128)]
    if backend != "python":
        axis_shapes.extend(((1024, 1024), (4096, 1024)))
    for rows, columns in axis_shapes:
        restriction = _ACCELERATED if rows > 128 else None
        values = ts.full((rows, columns), 2.0)

        def axis_sum(tensor=values) -> ts.Tensor:
            return ts.sum(tensor, axis=1)

        def validate_axis_sum(
            run=axis_sum,
            expected_rows=rows,
            expected_columns=columns,
        ) -> None:
            result = run()
            assert result.shape == (expected_rows,)
            assert result[0] == expected_columns * 2.0

        benchmarks.append(BenchmarkCase(
            name=f"scaling.axis_sum/{rows}x{columns}",
            run=axis_sum,
            validate=validate_axis_sum,
            work_items=rows * columns,
            description="public row-wise reduction scaling curve",
            backends=restriction,
        ))

    square_sizes = [16, 64]
    if backend != "python":
        square_sizes.extend((128, 256, 512, 1024))
    benchmarks.extend(
        _matmul_case(
            size,
            size,
            size,
            _ACCELERATED if size > 64 else None,
        )
        for size in square_sizes
    )
    if backend != "python":
        benchmarks.extend((
            _matmul_case(128, 256, 64, _ACCELERATED),
            _matmul_case(512, 1024, 256, _ACCELERATED),
        ))

    batch_sizes = [8]
    if backend != "python":
        batch_sizes.append(64)
    for batch in batch_sizes:
        restriction = _ACCELERATED if batch > 8 else None
        batched_left = ts.full((batch, 16, 16), 0.25)
        batched_right = ts.full((batch, 16, 16), 0.5)

        def batched_matmul(a=batched_left, b=batched_right) -> ts.Tensor:
            return ts.matmul(a, b)

        def validate_batched_matmul(
            run=batched_matmul,
            expected_batch=batch,
            expected_value=16 * 0.125,
        ) -> None:
            result = run()
            assert result.shape == (expected_batch, 16, 16)
            assert math.isclose(float(result[0, 0, 0]), expected_value)

        benchmarks.append(BenchmarkCase(
            name=f"scaling.batched_matmul/{batch}x16x16",
            run=batched_matmul,
            validate=validate_batched_matmul,
            work_items=batch * 16 ** 3,
            description="public broadcast-batched matrix multiplication curve",
            backends=restriction,
        ))
    return benchmarks
