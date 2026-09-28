"""Sound native certification and exact integer reduction primitives."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import numpy

from tensors.backend.numpy.conversion import _errstate

if TYPE_CHECKING:
    from tensors.dtype import DataType


def certified_float_sum(values: Any, axes: tuple[int, ...]) -> Any | None:
    """Return a correctly rounded native sum, or decline without guessing."""
    if not axes:
        return values
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        direct = numpy.sum(values, axis=axes, keepdims=True, dtype=values.dtype)
        nan = numpy.any(numpy.isnan(values), axis=axes, keepdims=True)
        positive_infinity = numpy.any(numpy.isposinf(values), axis=axes, keepdims=True)
        negative_infinity = numpy.any(numpy.isneginf(values), axis=axes, keepdims=True)
        finite = numpy.all(numpy.isfinite(values), axis=axes, keepdims=True)
        absolute = numpy.abs(values)
        nonzero = numpy.isfinite(values) & (values != 0.0)
        if values.dtype.itemsize == 8:
            unsigned_type = numpy.uint64
            fraction_mask = numpy.uint64((1 << 52) - 1)
            hidden_bit = numpy.uint64(1 << 52)
            exponent_shift = numpy.uint64(52)
            exponent_mask = numpy.uint64(0x7FF)
            exponent_bias = 1023
            fraction_bits = 52
            subnormal_exponent = -1074
        else:
            unsigned_type = numpy.uint32
            fraction_mask = numpy.uint32((1 << 23) - 1)
            hidden_bit = numpy.uint32(1 << 23)
            exponent_shift = numpy.uint32(23)
            exponent_mask = numpy.uint32(0xFF)
            exponent_bias = 127
            fraction_bits = 23
            subnormal_exponent = -149
        bits = absolute.view(unsigned_type)
        exponent_field = (bits >> exponent_shift) & exponent_mask
        mantissa = bits & fraction_mask
        mantissa = numpy.where(exponent_field == 0, mantissa, mantissa | hidden_bit)
        low_bit = mantissa & ((~mantissa) + unsigned_type(1))
        trailing_zeros = numpy.where(
            nonzero, numpy.log2(low_bit).astype(numpy.int64), 0
        )
        base_exponent = numpy.where(
            exponent_field == 0,
            subnormal_exponent,
            exponent_field.astype(numpy.int64) - exponent_bias - fraction_bits,
        )
        value_exponent = base_exponent + trailing_zeros
        quantum_exponent = numpy.min(
            numpy.where(nonzero, value_exponent, 4096),
            axis=axes,
            keepdims=True,
        )
        safe_exponent = numpy.where(quantum_exponent == 4096, 0, quantum_exponent)
        quantum = numpy.ldexp(
            numpy.ones_like(safe_exponent, dtype=values.dtype), safe_exponent
        )
        units = numpy.where(nonzero, absolute / quantum, 0.0)
        total_units = numpy.sum(units, axis=axes, keepdims=True, dtype=values.dtype)
        precision = numpy.finfo(values.dtype).nmant + 1
        exact_lattice = (total_units <= float(2**precision)) & numpy.isfinite(
            total_units * quantum
        )
        minimum = numpy.min(values, axis=axes, keepdims=True)
        maximum = numpy.max(values, axis=axes, keepdims=True)
        same_sign_overflow = numpy.isinf(direct) & ((minimum >= 0.0) | (maximum <= 0.0))
    nonfinite = nan | positive_infinity | negative_infinity
    at_most_one_addition = math.prod((values.shape[axis] for axis in axes)) <= 2
    finite_safe = at_most_one_addition | exact_lattice | same_sign_overflow
    certified = nonfinite | (finite & finite_safe)
    if not bool(numpy.all(certified)):
        return None
    both_infinities = positive_infinity & negative_infinity
    result = numpy.where(
        nan | both_infinities,
        numpy.nan,
        numpy.where(
            positive_infinity,
            numpy.inf,
            numpy.where(negative_infinity, -numpy.inf, direct),
        ),
    )
    return numpy.where(finite & (result == 0.0), 0.0, result)


def _uint64_sum(items: Any, axes: tuple[int, ...], keepdims: bool) -> tuple[Any, Any]:
    mask = numpy.uint64(0xFFFFFFFF)
    lower = numpy.sum(items & mask, axis=axes, keepdims=keepdims, dtype=numpy.uint64)
    upper = numpy.sum(
        items >> numpy.uint64(32),
        axis=axes,
        keepdims=keepdims,
        dtype=numpy.uint64,
    )
    upper = upper + (lower >> numpy.uint64(32))
    result = (upper << numpy.uint64(32)) | (lower & mask)
    return result, upper >> numpy.uint64(32)


def exact_integer_sum(
    values: Any,
    axes: tuple[int, ...],
    *,
    keepdims: bool,
    dtype: DataType,
) -> Any | None:
    """Sum integers exactly with two native uint32 limbs per magnitude."""
    count = math.prod((values.shape[axis] for axis in axes)) if axes else 1
    if count > 2**32:
        return None
    unsigned = values.astype(numpy.uint64, copy=False)
    if numpy.dtype(dtype.name).kind == "u":
        positive = unsigned
        negative = numpy.zeros_like(unsigned)
    else:
        positive = numpy.where(values > 0, unsigned, numpy.uint64(0))
        negative = numpy.where(
            values < 0, (~unsigned) + numpy.uint64(1), numpy.uint64(0)
        )
    positive_sum, positive_high = _uint64_sum(positive, axes, keepdims)
    negative_sum, negative_high = _uint64_sum(negative, axes, keepdims)
    positive_result = (positive_high > negative_high) | (
        (positive_high == negative_high) & (positive_sum >= negative_sum)
    )
    large_sum = numpy.where(positive_result, positive_sum, negative_sum)
    small_sum = numpy.where(positive_result, negative_sum, positive_sum)
    large_high = numpy.where(positive_result, positive_high, negative_high)
    small_high = numpy.where(positive_result, negative_high, positive_high)
    borrow = large_sum < small_sum
    with _errstate(over="ignore"):
        magnitude = large_sum - small_sum
        magnitude_high = large_high - small_high - borrow.astype(numpy.uint64)
    info = numpy.iinfo(numpy.dtype(dtype.name))
    if info.min == 0:
        overflow = (
            (magnitude_high != 0)
            | (negative_sum != 0)
            | (magnitude > numpy.uint64(info.max))
        )
        result = magnitude
    else:
        overflow = (magnitude_high != 0) | numpy.where(
            positive_result,
            magnitude > numpy.uint64(info.max),
            magnitude > numpy.uint64(-info.min),
        )
        with _errstate(over="ignore"):
            bits = numpy.where(
                positive_result, magnitude, (~magnitude) + numpy.uint64(1)
            )
        result = bits.view(numpy.int64)
    if bool(numpy.any(overflow)):
        raise OverflowError(f"integer sum exceeds {dtype.name}")
    return result.astype(numpy.dtype(dtype.name), copy=False)


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
    grouped = numpy.transpose(values, order).reshape(
        math.prod((input_shape[axis] for axis in remaining)),
        math.prod((input_shape[axis] for axis in axes)),
    )
    zero = numpy.any(grouped == 0, axis=1)
    unsigned = grouped.astype(numpy.uint64, copy=False)
    negative = grouped < 0
    magnitude = numpy.where(negative, (~unsigned) + numpy.uint64(1), unsigned)
    negative_result = numpy.count_nonzero(negative, axis=1) % 2 == 1
    factors = numpy.where(zero[:, None], numpy.uint64(1), magnitude)
    cumulative = numpy.cumprod(factors, axis=1, dtype=numpy.uint64)
    previous = numpy.concatenate(
        (
            numpy.ones((grouped.shape[0], 1), dtype=numpy.uint64),
            cumulative[:, :-1],
        ),
        axis=1,
    )
    overflow = numpy.any(cumulative // factors != previous, axis=1)
    accumulator = cumulative[:, -1]
    accumulator = numpy.where(zero, numpy.uint64(0), accumulator)
    if bool(numpy.any(overflow)):
        raise OverflowError(f"integer product exceeds {dtype.name}")
    info = numpy.iinfo(numpy.dtype(dtype.name))
    if info.min == 0:
        outside = negative_result | (accumulator > numpy.uint64(info.max))
        result = accumulator
    else:
        outside = numpy.where(
            negative_result,
            accumulator > numpy.uint64(-info.min),
            accumulator > numpy.uint64(info.max),
        )
        with _errstate(over="ignore"):
            bits = numpy.where(
                negative_result, (~accumulator) + numpy.uint64(1), accumulator
            )
        result = bits.view(numpy.int64)
    if bool(numpy.any(outside)):
        raise OverflowError(f"integer product exceeds {dtype.name}")
    return result.astype(numpy.dtype(dtype.name), copy=False).reshape(output_shape)
