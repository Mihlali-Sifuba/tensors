"""Deterministic pairwise floating summation for the CUDA backend.

The same tree as every other backend (`docs/summation-semantics.md`): each
reduction group in logical order, adjacent pairs added with one rounding to the
declared dtype, an odd final value carried unchanged, repeated until one value
remains. ``cupy.sum`` never performs the floating reduction, because its tree
is CuPy's; each round is one device-wide addition and only the O(log n) rounds
are iterated on the host. Nothing is read back to the host.

A ``float32`` group is reduced in binary32. CuPy's generated float32 code
flushes subnormals on this toolchain, so binary32 products and sums go through
the explicit round-to-nearest PTX instructions in
:mod:`tensors.backend.cuda.kernels.arithmetic.ieee32`, which keep gradual
underflow and cannot be contracted into a fused multiply-add. Binary64 already
underflows gradually on the device and uses CuPy's own operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import cupy

from tensors.backend.cuda.conversion import _errstate

if TYPE_CHECKING:
    from tensors.dtype import DataType


def _add(left: Any, right: Any) -> Any:
    """Add two arrays with one rounding to their shared dtype."""
    if left.dtype == cupy.float32:
        from tensors.backend.cuda.kernels.arithmetic import ieee32

        return ieee32.apply("add", left, right)
    return left + right


def _multiply(left: Any, right: Any) -> Any:
    """Multiply two arrays with one rounding to their shared dtype."""
    if left.dtype == cupy.float32:
        from tensors.backend.cuda.kernels.arithmetic import ieee32

        return ieee32.apply("multiply", left, right)
    return left * right


def to_declared_dtype(values: Any, dtype: DataType) -> Any:
    """Return ``values`` in the declared floating dtype, converting exactly.

    ``astype`` flushes a binary32 subnormal when it widens, so a crossing
    between formats goes through the PTX conversions in ``conversion``.
    Nothing here widens a ``float32`` computation.
    """
    from tensors.backend.cuda.conversion import _narrow, _widen

    target = cupy.dtype(dtype.name)
    values = cupy.asarray(values)
    if values.dtype == target:
        return values
    if target == cupy.float64:
        return _widen(values)
    return _narrow(_widen(values), target)


def _grouped(values: Any, axes: tuple[int, ...]) -> tuple[Any, tuple[int, ...]]:
    """Move the reduced axes last, in increasing order, and flatten them."""
    rank = values.ndim
    reduced = tuple(sorted(axis % rank for axis in axes))
    kept = tuple(axis for axis in range(rank) if axis not in reduced)
    moved = cupy.transpose(values, kept + reduced)
    group_size = 1
    for axis in reduced:
        group_size *= int(values.shape[axis])
    grouped = moved.reshape(tuple(int(values.shape[axis]) for axis in kept) + (group_size,))
    keepdims_shape = tuple(
        1 if axis in reduced else int(values.shape[axis]) for axis in range(rank)
    )
    return grouped, keepdims_shape


def _pairwise_last_axis(grouped: Any) -> Any:
    """Reduce the last axis with the specified adjacent-pair tree."""
    level = grouped
    while level.shape[-1] > 1:
        count = level.shape[-1]
        paired = count - count % 2
        summed = _add(level[..., 0:paired:2], level[..., 1:paired:2])
        if count % 2:
            summed = cupy.concatenate((summed, level[..., -1:]), axis=-1)
        level = summed
    return level[..., 0]


def pairwise_float_sum(values: Any, axes: tuple[int, ...]) -> Any:
    """Sum floating groups with the package's pairwise tree, keeping dims.

    Every addition rounds to ``values.dtype``. Non-finite groups are
    classified explicitly, and a zero result is canonical ``+0.0``.
    """
    if not axes:
        grouped = values[..., None]
        keepdims_shape = tuple(values.shape)
    else:
        grouped, keepdims_shape = _grouped(values, axes)
    if grouped.shape[-1] == 0:
        return cupy.zeros(keepdims_shape, dtype=values.dtype)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        tree = _pairwise_last_axis(grouped)
        nan = cupy.any(cupy.isnan(grouped), axis=-1)
        positive_infinity = cupy.any(cupy.isposinf(grouped), axis=-1)
        negative_infinity = cupy.any(cupy.isneginf(grouped), axis=-1)
        result = cupy.where(
            nan | (positive_infinity & negative_infinity),
            cupy.array(cupy.nan, dtype=values.dtype),
            cupy.where(
                positive_infinity,
                cupy.array(cupy.inf, dtype=values.dtype),
                cupy.where(
                    negative_infinity,
                    cupy.array(-cupy.inf, dtype=values.dtype),
                    tree,
                ),
            ),
        )
        result = cupy.where(
            _is_zero(result), cupy.zeros((), dtype=values.dtype), result
        )
    return cupy.asarray(result, dtype=values.dtype).reshape(keepdims_shape)


def _is_zero(values: Any) -> Any:
    """Whether each value is a signed zero, judged by its bit pattern.

    A float32 comparison against zero treats subnormals as zero on this
    toolchain, which would replace a subnormal result by ``+0.0``. Masking the
    sign bit and testing the remaining bits cannot.
    """
    if values.dtype == cupy.float32:
        return (values.view(cupy.uint32) & cupy.uint32(0x7FFFFFFF)) == 0
    return values == 0


__all__ = ["pairwise_float_sum", "to_declared_dtype"]
