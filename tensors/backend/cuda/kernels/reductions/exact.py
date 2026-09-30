"""Exact integer reduction primitives."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.conversion import _errstate

if TYPE_CHECKING:
    from tensors.dtype import DataType


def _uint64_sum(items: Any, axes: tuple[int, ...], keepdims: bool) -> tuple[Any, Any]:
    mask = cupy.uint64(0xFFFFFFFF)
    lower = cupy.sum(items & mask, axis=axes, keepdims=keepdims, dtype=cupy.uint64)
    upper = cupy.sum(
        items >> cupy.uint64(32),
        axis=axes,
        keepdims=keepdims,
        dtype=cupy.uint64,
    )
    upper = upper + (lower >> cupy.uint64(32))
    result = (upper << cupy.uint64(32)) | (lower & mask)
    return result, upper >> cupy.uint64(32)


def exact_integer_sum(
    values: Any,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Any | None:
    """Sum integers exactly with two device uint32 limbs per magnitude."""
    count = math.prod((values.shape[axis] for axis in axes)) if axes else 1
    if count > 2**32:
        return None
    unsigned = values.astype(cupy.uint64, copy=False)
    if cupy.dtype(dtype.name).kind == "u":
        positive = unsigned
        negative = cupy.zeros_like(unsigned)
    else:
        positive = cupy.where(values > 0, unsigned, cupy.uint64(0))
        negative = cupy.where(values < 0, (~unsigned) + cupy.uint64(1), cupy.uint64(0))
    positive_sum, positive_high = _uint64_sum(positive, axes, keepdims)
    negative_sum, negative_high = _uint64_sum(negative, axes, keepdims)
    positive_result = (positive_high > negative_high) | (
        (positive_high == negative_high) & (positive_sum >= negative_sum)
    )
    large_sum = cupy.where(positive_result, positive_sum, negative_sum)
    small_sum = cupy.where(positive_result, negative_sum, positive_sum)
    large_high = cupy.where(positive_result, positive_high, negative_high)
    small_high = cupy.where(positive_result, negative_high, positive_high)
    borrow = large_sum < small_sum
    with _errstate(over="ignore"):
        magnitude = large_sum - small_sum
        magnitude_high = large_high - small_high - borrow.astype(cupy.uint64)
    info = cupy.iinfo(cupy.dtype(dtype.name))
    if info.min == 0:
        overflow = (
            (magnitude_high != 0)
            | (negative_sum != 0)
            | (magnitude > cupy.uint64(info.max))
        )
        result = magnitude
    else:
        overflow = (magnitude_high != 0) | cupy.where(
            positive_result,
            magnitude > cupy.uint64(info.max),
            magnitude > cupy.uint64(-info.min),
        )
        with _errstate(over="ignore"):
            bits = cupy.where(positive_result, magnitude, (~magnitude) + cupy.uint64(1))
        result = bits.view(cupy.int64)
    if bool(cupy.any(overflow)):
        raise OverflowError(f"integer sum exceeds {dtype.name}")
    return result.astype(cupy.dtype(dtype.name), copy=False)


def exact_integer_product(
    values: Any,
    input_shape: tuple[int, ...],
    axes: tuple[int, ...],
    *,
    dtype: DataType,
    output_shape: tuple[int, ...],
) -> Any:
    """Multiply integer groups exactly until declared-range overflow is known."""
    remaining = tuple(axis for axis in range(len(input_shape)) if axis not in axes)
    order = remaining + axes
    grouped = cupy.transpose(values, order).reshape(
        math.prod((input_shape[axis] for axis in remaining)),
        math.prod((input_shape[axis] for axis in axes)),
    )
    zero = cupy.any(grouped == 0, axis=1)
    unsigned = grouped.astype(cupy.uint64, copy=False)
    negative = grouped < 0
    magnitude = cupy.where(negative, (~unsigned) + cupy.uint64(1), unsigned)
    negative_result = cupy.count_nonzero(negative, axis=1) % 2 == 1
    factors = cupy.where(zero[:, None], cupy.uint64(1), magnitude)
    cumulative = cupy.cumprod(factors, axis=1, dtype=cupy.uint64)
    previous = cupy.concatenate(
        (
            cupy.ones((grouped.shape[0], 1), dtype=cupy.uint64),
            cumulative[:, :-1],
        ),
        axis=1,
    )
    overflow = cupy.any(cumulative // factors != previous, axis=1)
    accumulator = cumulative[:, -1]
    accumulator = cupy.where(zero, cupy.uint64(0), accumulator)
    if bool(cupy.any(overflow)):
        raise OverflowError(f"integer product exceeds {dtype.name}")
    info = cupy.iinfo(cupy.dtype(dtype.name))
    if info.min == 0:
        outside = negative_result | (accumulator > cupy.uint64(info.max))
        result = accumulator
    else:
        outside = cupy.where(
            negative_result,
            accumulator > cupy.uint64(-info.min),
            accumulator > cupy.uint64(info.max),
        )
        with _errstate(over="ignore"):
            bits = cupy.where(
                negative_result, (~accumulator) + cupy.uint64(1), accumulator
            )
        result = bits.view(cupy.int64)
    if bool(cupy.any(outside)):
        raise OverflowError(f"integer product exceeds {dtype.name}")
    return result.astype(cupy.dtype(dtype.name), copy=False).reshape(output_shape)
