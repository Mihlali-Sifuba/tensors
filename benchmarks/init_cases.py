"""Parameter-initialization benchmark cases."""

from __future__ import annotations

import math

import tensors as ts

from .runner import BenchmarkBackend, BenchmarkCase


_ACCELERATED = frozenset[BenchmarkBackend]({"numpy", "cuda"})


def _initializer_case(
    name: str,
    shape: tuple[int, ...],
    initializer,
    work_items: int,
    description: str,
    backends: frozenset[BenchmarkBackend] | None,
) -> BenchmarkCase:
    def run() -> ts.Tensor:
        return initializer(shape)

    def validate() -> None:
        result = run()
        assert result.shape == shape
        assert all(
            math.isfinite(float(value)) for value in result._data[:4]
        )

    return BenchmarkCase(
        name=name,
        run=run,
        validate=validate,
        work_items=work_items,
        description=description,
        backends=backends,
    )


def cases() -> list[BenchmarkCase]:
    """Return one case per representative initializer family."""
    return [
        _initializer_case(
            "init.xavier_normal/10000",
            (100, 100),
            ts.init.xavier_normal,
            10_000,
            "Xavier normal initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.xavier_uniform/10000",
            (100, 100),
            ts.init.xavier_uniform,
            10_000,
            "Xavier uniform initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.he_normal/10000",
            (100, 100),
            ts.init.he_normal,
            10_000,
            "He normal initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.he_uniform/10000",
            (100, 100),
            ts.init.he_uniform,
            10_000,
            "He uniform initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.lecun_normal/10000",
            (100, 100),
            ts.init.lecun_normal,
            10_000,
            "LeCun normal initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.lecun_uniform/10000",
            (100, 100),
            ts.init.lecun_uniform,
            10_000,
            "LeCun uniform initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.variance_scaling/10000",
            (100, 100),
            ts.init.variance_scaling,
            10_000,
            "variance-scaling initialization with fan-in fan calculations",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.truncated_normal/10000",
            (100, 100),
            ts.init.truncated_normal,
            10_000,
            "truncated-normal initialization of a square matrix",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.orthogonal/64x64",
            (64, 64),
            ts.init.orthogonal,
            64 * 64,
            "orthogonal initialization with QR factorization",
            _ACCELERATED,
        ),
        _initializer_case(
            "init.fan_calculation/many_shapes",
            (128, 128),
            lambda shape: ts.init.he_uniform(shape),
            128 * 128,
            "he-uniform fan-in and fan-out calculation at a large shape",
            _ACCELERATED,
        ),
    ]
