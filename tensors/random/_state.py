"""MS-Tensors-owned random state and backend-native sampling."""

from __future__ import annotations

import importlib
import random as _stdlib_random
import threading
from array import array
from typing import Any

from ..backend import BackendName, get_backend
from ..dtype import DataType
from ..storage import CudaStorage, NumPyStorage, PythonStorage, Storage


_lock = threading.RLock()
_seed_value: int | None = None
_generators: dict[BackendName, Any] = {}


def seed(value: int | None) -> None:
    """Reset every backend stream to value on its next use."""
    global _seed_value
    with _lock:
        _seed_value = value
        _generators.clear()


def _generator(backend: BackendName) -> Any:
    generator = _generators.get(backend)
    if generator is not None:
        return generator
    if backend == "python":
        generator = _stdlib_random.Random(_seed_value)
    elif backend == "numpy":
        numpy = importlib.import_module("numpy")
        generator = numpy.random.default_rng(_seed_value)
    else:
        cupy = importlib.import_module("cupy")
        generator = cupy.random.RandomState(_seed_value)
    _generators[backend] = generator
    return generator


def _array_module(backend: BackendName) -> Any:
    return importlib.import_module("cupy" if backend == "cuda" else "numpy")


def _storage(values: Any, dtype: DataType, backend: BackendName) -> Storage:
    if backend == "python":
        return PythonStorage(array(dtype.typecode, values), dtype)
    module = _array_module(backend)
    contiguous = module.asarray(
        values,
        dtype=module.dtype(dtype.name),
    ).reshape(-1)
    if backend == "cuda":
        return CudaStorage(contiguous, dtype)
    return NumPyStorage(contiguous, dtype)


def uniform(count: int, low: float, high: float, dtype: DataType) -> Storage:
    """Draw a flat uniform sample on the active backend."""
    backend = get_backend()
    with _lock:
        generator = _generator(backend)
        if backend == "python":
            width = high - low
            values = (low + width * generator.random() for _ in range(count))
        else:
            values = generator.uniform(low, high, size=count)
        return _storage(values, dtype, backend)


def normal(count: int, mean: float, stddev: float, dtype: DataType) -> Storage:
    """Draw a flat normal sample on the active backend."""
    backend = get_backend()
    with _lock:
        generator = _generator(backend)
        if backend == "python":
            values = (generator.gauss(mean, stddev) for _ in range(count))
        else:
            values = generator.normal(mean, stddev, size=count)
        return _storage(values, dtype, backend)


def randint(count: int, low: int, high: int, dtype: DataType) -> Storage:
    """Draw flat integers from the half-open interval [low, high)."""
    backend = get_backend()
    with _lock:
        generator = _generator(backend)
        if backend == "python":
            values = (generator.randrange(low, high) for _ in range(count))
        elif backend == "numpy":
            values = generator.integers(low, high, size=count)
        else:
            values = generator.randint(low, high, size=count)
        return _storage(values, dtype, backend)


def truncated_normal(
    count: int,
    mean: float,
    stddev: float,
    lower: float,
    upper: float,
    dtype: DataType,
) -> Storage:
    """Draw a flat bounded normal sample by backend-native rejection."""
    backend = get_backend()
    with _lock:
        generator = _generator(backend)
        if count == 0:
            return _storage([], dtype, backend)
        if backend == "python":
            values: list[float] = []
            while len(values) < count:
                candidate = generator.gauss(mean, stddev)
                if lower <= candidate <= upper:
                    values.append(candidate)
            return _storage(values, dtype, backend)

        module = _array_module(backend)
        values = generator.normal(mean, stddev, size=count)
        mask = (values < lower) | (values > upper)
        while bool(module.any(mask)):
            rejected = int(module.count_nonzero(mask))
            values[mask] = generator.normal(mean, stddev, size=rejected)
            mask = (values < lower) | (values > upper)
        return _storage(values, dtype, backend)


__all__ = ["normal", "randint", "seed", "truncated_normal", "uniform"]
