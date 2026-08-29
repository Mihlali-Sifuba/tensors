"""Truncated-normal and orthogonal parameter initializers."""

from __future__ import annotations

import importlib
import math
from collections.abc import Iterable
from typing import TypeAlias

from ..backend import get_backend
from ..dtype import DataType
from ..random import normal
from ..random import _truncated_normal
from ..storage import CudaStorage, NumPyStorage, PythonStorage, Storage
from ..tensor import Tensor
from ..utils.shape import normalize_shape, shape_size
from ._variance import _finite_number, _floating_dtype


Shape: TypeAlias = Iterable[int]
DType: TypeAlias = str | DataType | None


def truncated_normal(
    shape: Shape,
    mean: int | float = 0.0,
    stddev: int | float = 1.0,
    lower: int | float | None = None,
    upper: int | float | None = None,
    dtype: DType = None,
) -> Tensor:
    """Sample a normal distribution subject to inclusive absolute bounds.

    stddev describes the source normal before truncation. Omitted bounds
    default to two source standard deviations on either side of mean.
    """
    center = _finite_number("mean", mean)
    deviation = _finite_number("stddev", stddev)
    if deviation <= 0.0:
        raise ValueError("stddev must be greater than zero")
    minimum = center - 2.0 * deviation if lower is None else lower
    maximum = center + 2.0 * deviation if upper is None else upper
    return _truncated_normal(
        shape,
        mean=center,
        stddev=deviation,
        lower=minimum,
        upper=maximum,
        dtype=dtype,
    )


def _python_orthogonal(
    rows: int,
    columns: int,
    dtype: DataType,
    gain: float,
) -> Storage:
    source_rows = max(rows, columns)
    source_columns = min(rows, columns)
    source = normal((source_rows, source_columns), dtype=dtype)
    raw = source.tolist()
    vectors: list[list[float]] = []
    for column in range(source_columns):
        vector = [
            float(raw[row * source_columns + column])
            for row in range(source_rows)
        ]
        for basis in vectors:
            projection = math.fsum(
                value * basis_value
                for value, basis_value in zip(vector, basis)
            )
            vector = [
                value - projection * basis_value
                for value, basis_value in zip(vector, basis)
            ]
        magnitude = math.sqrt(math.fsum(value * value for value in vector))
        if magnitude <= 1e-15:
            raise RuntimeError(
                "orthogonal initialization encountered a degenerate sample"
            )
        vectors.append([value / magnitude for value in vector])
    matrix = [
        vectors[column][row]
        for row in range(source_rows)
        for column in range(source_columns)
    ]
    if rows < columns:
        matrix = [
            matrix[column * rows + row]
            for row in range(rows)
            for column in range(columns)
        ]
    return PythonStorage.from_values(
        (gain * value for value in matrix),
        dtype,
    )


def _array_orthogonal(
    rows: int,
    columns: int,
    dtype: DataType,
    gain: float,
) -> Storage:
    backend = get_backend()
    module = importlib.import_module("cupy" if backend == "cuda" else "numpy")
    source = normal((rows, columns), dtype=dtype)
    matrix = source._storage.buffer.reshape(rows, columns)
    transposed = rows < columns
    if transposed:
        matrix = matrix.T
    q, r = module.linalg.qr(matrix, mode="reduced")
    signs = module.sign(module.diag(r))
    signs = module.where(signs == 0, 1, signs)
    q = q * signs
    if transposed:
        q = q.T
    values = module.asarray(
        q * gain,
        dtype=module.dtype(dtype.name),
    ).reshape(-1)
    if backend == "cuda":
        return CudaStorage(values, dtype)
    return NumPyStorage(values, dtype)


def orthogonal(
    shape: Shape,
    gain: int | float = 1.0,
    dtype: DType = None,
) -> Tensor:
    """Return a tensor whose flattened rows or columns are orthogonal.

    The tensor is viewed as (shape[0], product(shape[1:])). The smaller
    dimension is a complete orthonormal basis, scaled by gain.
    """
    normalized_shape = normalize_shape(shape)
    if len(normalized_shape) < 2:
        raise ValueError(
            "orthogonal initialization requires at least two dimensions"
        )
    if any(dimension == 0 for dimension in normalized_shape):
        raise ValueError(
            "orthogonal initialization requires positive dimensions"
        )
    multiplier = _finite_number("gain", gain)
    resolved_dtype = _floating_dtype(dtype)
    rows = normalized_shape[0]
    columns = math.prod(normalized_shape[1:])
    if get_backend() == "python":
        storage = _python_orthogonal(
            rows, columns, resolved_dtype, multiplier
        )
    else:
        storage = _array_orthogonal(
            rows, columns, resolved_dtype, multiplier
        )
    if storage.size != shape_size(normalized_shape):
        raise RuntimeError(
            "orthogonal initializer returned an unexpected result size"
        )
    return Tensor(storage, dtype=resolved_dtype, shape=normalized_shape)


__all__ = ["orthogonal", "truncated_normal"]
