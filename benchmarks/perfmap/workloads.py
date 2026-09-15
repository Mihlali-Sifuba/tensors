"""Shared size curves, dtype selections, and input builders.

Every suite draws its shapes and dtypes from here so that a curve means the
same thing across suites, and so that the per-backend ceilings are stated
once. The Python backend interprets element by element, and this machine's
GPU has roughly 3.4 GB free, so both need explicit ceilings rather than a
size curve that happens not to fall over.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.shape import Shape

from .harness import Unsupported


#: Accelerated backends share the NumPy/CuPy kernel implementations.
ACCELERATED = frozenset({"numpy", "cuda"})
ALL_BACKENDS = frozenset({"python", "numpy", "cuda"})
CUDA_ONLY = frozenset({"cuda"})
NUMPY_ONLY = frozenset({"numpy"})

#: The systematic elementwise curve. Powers of ten expose fixed overhead at
#: the bottom and asymptotic throughput at the top.
ELEMENTWISE_SIZES: tuple[int, ...] = (
    1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000,
)

#: A shorter curve for paths too expensive to sweep at full resolution.
COARSE_SIZES: tuple[int, ...] = (1, 100, 10_000, 1_000_000)

#: The largest element count each backend can carry without dominating the
#: run or exhausting device memory. Kernels widen floating point to float64
#: internally, so a CUDA element costs 8 bytes plus temporaries whatever the
#: tensor dtype is.
SIZE_CEILING: dict[str, int] = {
    "python": 100_000,
    "numpy": 10_000_000,
    "cuda": 10_000_000,
}

#: Reductions build several full-size temporaries inside the stability
#: guard, so their ceiling is lower than plain elementwise work.
REDUCTION_CEILING: dict[str, int] = {
    "python": 100_000,
    "numpy": 10_000_000,
    "cuda": 4_000_000,
}

#: Differentiated paths hold forward values plus gradients, so they are
#: capped lower again.
GRADIENT_CEILING: dict[str, int] = {
    "python": 10_000,
    "numpy": 1_000_000,
    "cuda": 1_000_000,
}

#: Square matrix side lengths. The small end is where framework overhead
#: dominates; the large end is where the provider's GEMM does.
MATRIX_SIDES: dict[str, tuple[int, ...]] = {
    "python": (2, 8, 32),
    "numpy": (2, 8, 32, 128, 512, 1024, 2048),
    "cuda": (2, 8, 32, 128, 512, 1024, 2048),
}

#: Rectangular matrix shapes as ``(rows, contraction, columns)``.
RECTANGULAR_SHAPES: tuple[tuple[int, int, int], ...] = (
    (1, 1024, 1024),
    (1024, 1024, 1),
    (8, 4096, 8),
    (256, 2048, 64),
    (1024, 16, 1024),
)

#: The dtypes worth separating. ``float32`` and ``float64`` differ in both
#: hardware behavior and library handling; integers and bool exercise
#: different code paths entirely.
FLOAT_DTYPES: tuple[str, ...] = ("float64", "float32")
INTEGER_DTYPES: tuple[str, ...] = ("int64", "int32")
NUMERIC_DTYPES: tuple[str, ...] = ("float64", "float32", "int64", "int32")


def dtype_of(name: str) -> Any:
    """Return the package DataType for a dtype name."""
    return getattr(ts, name)


def is_integer(dtype_name: str) -> bool:
    """Return whether a dtype name denotes an integer type."""
    return dtype_of(dtype_name).kind == "integer"


def provider_module(backend: str) -> Any:
    """Return the raw array provider backing ``backend``."""
    if backend == "cuda":
        return importlib.import_module("cupy")
    if backend == "numpy":
        return importlib.import_module("numpy")
    raise Unsupported(
        "the Python backend has no array provider to compare against; its "
        "reference kernels are the implementation"
    )


def kernel_module(backend: str) -> Any:
    """Return the internal kernel facade the loader resolves for ``backend``."""
    if backend not in ACCELERATED:
        raise Unsupported(
            "internal array kernels exist only for the NumPy and CUDA "
            "backends; the Python backend is the reference implementation"
        )
    return importlib.import_module(f"tensors.backend.{backend}")


def sizes_for(
    backend: str,
    *,
    ceiling: dict[str, int] | None = None,
    curve: Sequence[int] = ELEMENTWISE_SIZES,
) -> tuple[int, ...]:
    """Return the part of a size curve ``backend`` can carry."""
    limit = (ceiling or SIZE_CEILING)[backend]
    return tuple(size for size in curve if size <= limit)


def matrix_sides(backend: str) -> tuple[int, ...]:
    """Return the square matrix sides ``backend`` can carry."""
    return MATRIX_SIDES[backend]


# -- value builders ---------------------------------------------------
#
# Constant data is same-sign and finite, which is the case the reduction
# stability guard has a fast path for. Mixed-magnitude, mixed-sign data
# takes the full guard instead. Both are benchmarked, so the builders are
# separate and named for what they exercise rather than for their values.


def constant_values(size: int, value: float, dtype_name: str) -> list[Any]:
    """Return ``size`` copies of one value in the right Python type."""
    if is_integer(dtype_name):
        return [int(value)] * size
    return [float(value)] * size


def ramp_values(size: int, dtype_name: str) -> list[Any]:
    """Return finite, same-sign, non-degenerate values.

    A ramp avoids the pathologies of an all-identical buffer without leaving
    the ordinary numeric range any guard is tuned for.
    """
    if is_integer(dtype_name):
        return [(index % 97) + 1 for index in range(size)]
    return [1.0 + (index % 97) / 97.0 for index in range(size)]


def mixed_sign_values(size: int, dtype_name: str) -> list[Any]:
    """Return alternating-sign values that defeat the same-sign fast path."""
    if is_integer(dtype_name):
        return [((index % 97) + 1) * (1 if index % 2 else -1) for index in range(size)]
    return [
        (1.0 + (index % 97) / 97.0) * (1.0 if index % 2 else -1.0)
        for index in range(size)
    ]


def tensor(
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    value: float = 1.5,
) -> ts.Tensor:
    """Build a Tensor of ``shape`` holding storage native to the backend.

    ``kind`` selects the value pattern: ``constant`` for a filled buffer,
    ``ramp`` for finite same-sign variety, and ``mixed`` for alternating
    signs that take the full stability guard.

    The public constructor accepts only host values, so building a large
    input through it would both dominate setup and leave the tensor holding
    host storage that a real workload would not have. Inputs are therefore
    assembled in the backend's own representation, which is the state
    ``zeros``, ``full`` and every accelerated operation produce.
    """
    size = math.prod(shape) if shape else 1
    dtype = dtype_of(dtype_name)
    if kind == "constant":
        return ts.full(
            shape,
            int(value) if dtype.kind == "integer" else value,
            dtype=dtype,
        )

    backend = ts.get_backend()
    if backend == "python":
        values = (
            ramp_values(size, dtype_name)
            if kind == "ramp"
            else mixed_sign_values(size, dtype_name)
        )
        return ts.Tensor(values, dtype=dtype, shape=shape)

    provider = provider_module(backend)
    native = _pattern_array(provider, size, dtype_name, kind)
    storage = _native_storage(backend, native, dtype)
    return ts.Tensor._from_owned_storage(
        storage, dtype=dtype, shape=Shape.from_iterable(shape)
    )


def _pattern_array(
    provider: Any, size: int, dtype_name: str, kind: str
) -> Any:
    """Build the value pattern with provider arithmetic rather than a list."""
    native_dtype = getattr(provider, dtype_name)
    index = provider.arange(size, dtype=provider.int64)
    if is_integer(dtype_name):
        magnitude = (index % 97) + 1
    else:
        magnitude = 1.0 + (index % 97) / 97.0
    if kind == "mixed":
        signs = provider.where((index % 2) == 1, 1, -1)
        magnitude = magnitude * signs
    elif kind != "ramp":
        raise ValueError(f"unknown value kind {kind!r}")
    return magnitude.astype(native_dtype)


def _native_storage(backend: str, native: Any, dtype: Any) -> Any:
    """Wrap a provider array in the storage class for ``backend``."""
    from tensors.backend.storage import CudaStorage, NumPyStorage

    if backend == "cuda":
        return CudaStorage(native, dtype)
    return NumPyStorage(native, dtype)


def provider_array(
    provider: Any,
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    value: float = 1.5,
) -> Any:
    """Build the provider-native array matching :func:`tensor`."""
    size = math.prod(shape) if shape else 1
    native_dtype = getattr(provider, dtype_name)
    if kind == "constant":
        return provider.full(shape, value, dtype=native_dtype)
    return _pattern_array(provider, size, dtype_name, kind).reshape(shape)


def variable(
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    requires_grad: bool = True,
) -> ts.Variable:
    """Build a Variable wrapping :func:`tensor`."""
    return ts.Variable(
        tensor(shape, dtype_name=dtype_name, kind=kind),
        requires_grad=requires_grad,
    )


# -- validation helpers ------------------------------------------------


def scalar(value: Any) -> float:
    """Return a Python float from a Tensor, Storage buffer, array, or number."""
    if isinstance(value, ts.Tensor):
        return float(value.tolist()[0]) if value.size else math.nan
    item = getattr(value, "item", None)
    if callable(item):
        return float(item())
    return float(value)


def first(value: Any) -> float:
    """Return the first logical element of a Tensor or provider array."""
    if isinstance(value, ts.Tensor):
        return float(value.tolist()[0])
    flat = value.reshape(-1) if hasattr(value, "reshape") else value
    return float(flat[0])


def close(left: float, right: float, *, tolerance: float = 1e-6) -> bool:
    """Return whether two scalars agree within a relative tolerance."""
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def label(shape: tuple[int, ...]) -> str:
    """Return a compact shape label usable inside a case name."""
    return "x".join(str(dimension) for dimension in shape) or "scalar"


__all__ = [
    "ACCELERATED",
    "ALL_BACKENDS",
    "COARSE_SIZES",
    "CUDA_ONLY",
    "ELEMENTWISE_SIZES",
    "FLOAT_DTYPES",
    "GRADIENT_CEILING",
    "INTEGER_DTYPES",
    "MATRIX_SIDES",
    "NUMERIC_DTYPES",
    "NUMPY_ONLY",
    "RECTANGULAR_SHAPES",
    "REDUCTION_CEILING",
    "SIZE_CEILING",
    "close",
    "constant_values",
    "dtype_of",
    "first",
    "is_integer",
    "kernel_module",
    "label",
    "matrix_sides",
    "mixed_sign_values",
    "provider_array",
    "provider_module",
    "ramp_values",
    "scalar",
    "sizes_for",
    "tensor",
    "variable",
]
