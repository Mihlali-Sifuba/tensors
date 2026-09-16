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
from benchmarks import profiles
from benchmarks import baselines
from benchmarks.baselines import _native as native
from benchmarks.case import Unsupported

ACCELERATED = frozenset({"numpy", "cuda"})
ALL_BACKENDS = frozenset({"python", "numpy", "cuda"})
CUDA_ONLY = frozenset({"cuda"})
NUMPY_ONLY = frozenset({"numpy"})
ELEMENTWISE_SIZES: tuple[int, ...] = (
    1,
    10,
    100,
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
)
COARSE_SIZES: tuple[int, ...] = (1, 100, 10_000, 1_000_000)
SIZE_CEILING: dict[str, int] = {
    "python": 100_000,
    "numpy": 10_000_000,
    "cuda": 10_000_000,
}
REDUCTION_CEILING: dict[str, int] = {
    "python": 100_000,
    "numpy": 10_000_000,
    "cuda": 4_000_000,
}
GRADIENT_CEILING: dict[str, int] = {
    "python": 10_000,
    "numpy": 1_000_000,
    "cuda": 1_000_000,
}
MATRIX_SIDES: dict[str, tuple[int, ...]] = {
    "python": (2, 8, 32),
    "numpy": (2, 8, 32, 128, 512, 1_024, 2_048),
    "cuda": (2, 8, 32, 128, 512, 1_024, 2_048),
}
RECTANGULAR_SHAPES: tuple[tuple[int, int, int], ...] = (
    (1, 1_024, 1_024),
    (1_024, 1_024, 1),
    (8, 4_096, 8),
    (256, 2_048, 64),
    (1_024, 16, 1_024),
)
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
    """Return the external library ``backend`` is measured against.

    Defined in :mod:`benchmarks.baselines`, so that "raw NumPy" means the
    same thing to every suite that compares against it.
    """
    return baselines.module(backend)


def kernel_module(backend: str) -> Any:
    """Return the internal kernel facade the loader resolves for ``backend``."""
    if backend not in ACCELERATED:
        raise Unsupported(
            "internal array kernels exist only for the NumPy and CUDA backends; the Python backend is the reference implementation"
        )
    return importlib.import_module(f"tensors.backend.{backend}.kernels")


def ceiling_for(backend: str, table: dict[str, int] | None = None) -> int:
    """Return the element ceiling in force for ``backend``.

    A workload states the ceiling its own cost imposes; a profile may state a
    tighter one for the run. The smaller of the two wins, so neither can be
    talked out of a limit by the other.
    """
    declared = (table or SIZE_CEILING)[backend]
    return profiles.active().ceiling_for(backend, declared)


def sizes_for(
    backend: str,
    *,
    ceiling: dict[str, int] | None = None,
    curve: Sequence[int] = ELEMENTWISE_SIZES,
) -> tuple[int, ...]:
    """Return the part of a size curve this backend and profile admit."""
    limit = ceiling_for(backend, ceiling)
    return tuple(size for size in profiles.selected_sizes(curve) if size <= limit)


def dtypes_for(names: Sequence[str]) -> tuple[str, ...]:
    """Return the part of a dtype selection the active profile admits."""
    return profiles.selected_dtypes(names)


def matrix_sides(backend: str) -> tuple[int, ...]:
    """Return the square matrix sides ``backend`` can carry."""
    return MATRIX_SIDES[backend]


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
        return [index % 97 + 1 for index in range(size)]
    return [1.0 + index % 97 / 97.0 for index in range(size)]


def mixed_sign_values(size: int, dtype_name: str) -> list[Any]:
    """Return alternating-sign values that defeat the same-sign fast path."""
    if is_integer(dtype_name):
        return [(index % 97 + 1) * (1 if index % 2 else -1) for index in range(size)]
    return [
        (1.0 + index % 97 / 97.0) * (1.0 if index % 2 else -1.0)
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
            shape, int(value) if dtype.kind == "integer" else value, dtype=dtype
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
    # The same builder the baselines use, so a Tensor and the array it is
    # compared against hold identical values rather than merely similar ones.
    buffer = native.pattern(provider, size, dtype_name, kind)
    storage = _native_storage(backend, buffer, dtype)
    return ts.Tensor._from_owned_storage(
        storage, dtype=dtype, shape=Shape.from_iterable(shape)
    )


def _native_storage(backend: str, native: Any, dtype: Any) -> Any:
    """Wrap a provider array in the storage class for ``backend``."""
    from tensors.backend.cuda.storage import CudaStorage
    from tensors.backend.numpy.storage import NumPyStorage

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
    return native.build(provider, shape, dtype_name=dtype_name, kind=kind, value=value)


def variable(
    shape: tuple[int, ...],
    *,
    dtype_name: str = "float64",
    kind: str = "ramp",
    requires_grad: bool = True,
) -> ts.Variable:
    """Build a Variable wrapping :func:`tensor`."""
    return ts.Variable(
        tensor(shape, dtype_name=dtype_name, kind=kind), requires_grad=requires_grad
    )


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


def close(left: float, right: float, *, tolerance: float = 1e-06) -> bool:
    """Return whether two scalars agree within a relative tolerance."""
    return math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)


def label(shape: tuple[int, ...]) -> str:
    """Return a compact shape label usable inside a case name."""
    return "x".join((str(dimension) for dimension in shape)) or "scalar"


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
    "ceiling_for",
    "dtypes_for",
    "sizes_for",
    "tensor",
    "variable",
]
