"""Deterministic pairwise floating summation for the NumPy backend.

`docs/summation-semantics.md` specifies the reduction tree: each reduction
group is taken in logical order, adjacent pairs are added with one rounding to
the declared dtype, an odd final value is carried unchanged into the next
round, and the rounds repeat until one value remains. The tree belongs to the
package, not to NumPy, so ``numpy.sum`` never performs the floating reduction
here: each round is one vectorised addition over every group at once, and only
the O(log n) rounds are iterated in Python.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy

from tensors.backend.numpy.conversion import _errstate

if TYPE_CHECKING:
    from tensors.dtype import DataType


def to_declared_dtype(values: Any, dtype: DataType) -> Any:
    """Return ``values`` in the declared floating dtype, converting exactly.

    Summation and contraction run in the declared dtype. Widening a binary32
    operand to binary64 is exact; nothing here widens a ``float32``
    computation.
    """
    return numpy.asarray(values).astype(numpy.dtype(dtype.name), copy=False)


def _grouped(values: Any, axes: tuple[int, ...]) -> tuple[Any, tuple[int, ...]]:
    """Move the reduced axes last, in increasing order, and flatten them.

    The group sequence is the reduced coordinates in row-major order over the
    reduced axes taken in increasing axis order. Returns the grouped array and
    the keepdims shape the result is restored to.
    """
    rank = values.ndim
    reduced = tuple(sorted(axis % rank for axis in axes))
    kept = tuple(axis for axis in range(rank) if axis not in reduced)
    moved = numpy.transpose(values, kept + reduced)
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
        summed = level[..., 0:paired:2] + level[..., 1:paired:2]
        if count % 2:
            summed = numpy.concatenate((summed, level[..., -1:]), axis=-1)
        level = summed
    return level[..., 0]


def pairwise_float_sum(values: Any, axes: tuple[int, ...]) -> Any:
    """Sum floating groups with the package's pairwise tree, keeping dims.

    Every addition rounds to ``values.dtype``: a ``float32`` group is reduced
    in binary32 and a ``float64`` group in binary64. Non-finite groups are
    classified explicitly, and a zero result is canonical ``+0.0``.
    """
    if not axes:
        grouped = values[..., None]
        keepdims_shape = tuple(values.shape)
    else:
        grouped, keepdims_shape = _grouped(values, axes)
    if grouped.shape[-1] == 0:
        return numpy.zeros(keepdims_shape, dtype=values.dtype)
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        tree = _pairwise_last_axis(grouped)
        nan = numpy.any(numpy.isnan(grouped), axis=-1)
        positive_infinity = numpy.any(numpy.isposinf(grouped), axis=-1)
        negative_infinity = numpy.any(numpy.isneginf(grouped), axis=-1)
        result = numpy.where(
            nan | (positive_infinity & negative_infinity),
            numpy.array(numpy.nan, dtype=values.dtype),
            numpy.where(
                positive_infinity,
                numpy.array(numpy.inf, dtype=values.dtype),
                numpy.where(
                    negative_infinity,
                    numpy.array(-numpy.inf, dtype=values.dtype),
                    tree,
                ),
            ),
        )
        result = numpy.where(result == 0, numpy.zeros((), dtype=values.dtype), result)
    return numpy.asarray(result, dtype=values.dtype).reshape(keepdims_shape)


__all__ = ["pairwise_float_sum", "to_declared_dtype"]
