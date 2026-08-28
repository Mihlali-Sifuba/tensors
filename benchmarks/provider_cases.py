"""Raw provider and internal-kernel controls for overhead attribution."""

from __future__ import annotations

import importlib
import math
from typing import Any

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _scalar(value: Any) -> float:
    item = value.item() if hasattr(value, "item") else value
    return float(item)


def cases() -> list[BenchmarkCase]:
    """Compare provider primitives with tensors' guarded native kernels."""
    backend = ts.get_backend()
    if backend not in _ACCELERATED:
        return []

    provider = importlib.import_module("cupy" if backend == "cuda" else "numpy")
    kernels = importlib.import_module(f"tensors.backend.{backend}")
    benchmarks: list[BenchmarkCase] = []

    for size in (100, 10_000, 100_000, 1_000_000, 10_000_000):
        raw_left = provider.full(size, 1.5, dtype=provider.float64)
        raw_right = provider.full(size, 2.0, dtype=provider.float64)
        tensor_left = ts.full((size,), 1.5)
        tensor_right = ts.full((size,), 2.0)

        def raw_add(left=raw_left, right=raw_right):
            return provider.add(left, right)

        def validate_raw_add(run=raw_add, expected_size=size) -> None:
            result = run()
            assert result.size == expected_size
            assert _scalar(result.reshape(-1)[-1]) == 3.5

        benchmarks.append(BenchmarkCase(
            name=f"provider.add/{size}",
            run=raw_add,
            validate=validate_raw_add,
            work_items=size,
            description="raw provider elementwise addition",
            layer="provider",
            backends=_ACCELERATED,
        ))

        def kernel_add(left=tensor_left, right=tensor_right, output_size=size):
            return kernels.binary(
                "add",
                left,
                right,
                dtype=ts.float64,
                output_shape=(output_size,),
            )

        def validate_kernel_add(run=kernel_add, expected_size=size) -> None:
            storage = run()
            assert storage is not None
            assert storage.size == expected_size
            assert _scalar(storage.buffer[-1]) == 3.5

        benchmarks.append(BenchmarkCase(
            name=f"kernel.add/{size}",
            run=kernel_add,
            validate=validate_kernel_add,
            work_items=size,
            description="guarded tensors backend addition kernel",
            layer="kernel",
            backends=_ACCELERATED,
        ))

        def raw_sum(values=raw_left):
            return provider.sum(values)

        def validate_raw_sum(run=raw_sum, expected_size=size) -> None:
            assert math.isclose(_scalar(run()), expected_size * 1.5)

        benchmarks.append(BenchmarkCase(
            name=f"provider.sum/{size}",
            run=raw_sum,
            validate=validate_raw_sum,
            work_items=size,
            description="raw provider full reduction",
            layer="provider",
            backends=_ACCELERATED,
        ))

        def kernel_sum(values=tensor_left):
            return kernels.reduction(
                "sum",
                values,
                (0,),
                keepdims=False,
                dtype=ts.float64,
                output_shape=(1,),
            )

        def validate_kernel_sum(run=kernel_sum, expected_size=size) -> None:
            storage = run()
            assert storage is not None
            assert math.isclose(
                _scalar(storage.buffer[0]),
                expected_size * 1.5,
            )

        benchmarks.append(BenchmarkCase(
            name=f"kernel.sum/{size}",
            run=kernel_sum,
            validate=validate_kernel_sum,
            work_items=size,
            description="guarded tensors backend full reduction",
            layer="kernel",
            backends=_ACCELERATED,
        ))

    for size in (16, 64, 128, 256, 512, 1024):
        raw_left = provider.full((size, size), 0.25, dtype=provider.float64)
        raw_right = provider.full((size, size), 0.5, dtype=provider.float64)
        tensor_left = ts.full((size, size), 0.25)
        tensor_right = ts.full((size, size), 0.5)
        expected = size * 0.125

        def raw_matmul(left=raw_left, right=raw_right):
            return provider.matmul(left, right)

        def validate_raw_matmul(run=raw_matmul, expected_value=expected) -> None:
            result = run()
            assert math.isclose(_scalar(result[0, 0]), expected_value)

        benchmarks.append(BenchmarkCase(
            name=f"provider.matmul/{size}x{size}",
            run=raw_matmul,
            validate=validate_raw_matmul,
            work_items=size ** 3,
            description="raw provider square matrix multiplication",
            layer="provider",
            backends=_ACCELERATED,
        ))

        def kernel_matmul(
            left=tensor_left,
            right=tensor_right,
            output_size=size,
        ):
            return kernels.matmul(
                left,
                right,
                dtype=ts.float64,
                output_shape=(output_size, output_size),
            )

        def validate_kernel_matmul(
            run=kernel_matmul,
            expected_value=expected,
        ) -> None:
            storage = run()
            assert storage is not None
            assert math.isclose(_scalar(storage.buffer[0]), expected_value)

        benchmarks.append(BenchmarkCase(
            name=f"kernel.matmul/{size}x{size}",
            run=kernel_matmul,
            validate=validate_kernel_matmul,
            work_items=size ** 3,
            description="guarded tensors backend matrix multiplication kernel",
            layer="kernel",
            backends=_ACCELERATED,
        ))

    return benchmarks
