"""Sound native certification and exact integer reduction primitives."""

from __future__ import annotations

import math
from typing import Any, TYPE_CHECKING

import cupy

from tensors.backend.cuda.conversion import _errstate
from tensors.backend.cuda.conversion import _narrow
from tensors.backend.cuda.conversion import _widen

if TYPE_CHECKING:
    from tensors.dtype import DataType


def _compensated_candidate(
    values: Any, axes: tuple[int, ...], target_dtype: Any
) -> tuple[Any, Any]:
    """Enclose the exact sum of finite binary64 groups and certify rounding."""
    remaining = tuple(axis for axis in range(values.ndim) if axis not in axes)
    order = remaining + axes
    group_shape = tuple(values.shape[axis] for axis in remaining)
    kept_shape = tuple(
        1 if axis in axes else values.shape[axis] for axis in range(values.ndim)
    )
    grouped = cupy.transpose(values, order).reshape(
        math.prod(group_shape), math.prod((values.shape[axis] for axis in axes))
    )
    finite = cupy.all(cupy.isfinite(grouped), axis=1)
    grouped = cupy.where(cupy.isfinite(grouped), grouped, 0.0)
    total = cupy.zeros(grouped.shape[0], dtype=values.dtype)
    lower_error = cupy.zeros_like(total)
    upper_error = cupy.zeros_like(total)
    valid = finite.copy()
    with _errstate(over="ignore", under="ignore", invalid="ignore"):
        for index in range(grouped.shape[1]):
            item = grouped[:, index]
            updated = total + item
            virtual_item = updated - total
            virtual_total = updated - virtual_item
            error = (total - virtual_total) + (item - virtual_item)
            valid &= cupy.isfinite(updated) & cupy.isfinite(error)
            lower_error = cupy.nextafter(lower_error + error, -cupy.inf)
            upper_error = cupy.nextafter(upper_error + error, cupy.inf)
            total = updated
        midpoint_error = lower_error + (upper_error - lower_error) * 0.5
        approximate = total + midpoint_error
        candidate_native = _narrow(approximate, target_dtype)
        candidate = _widen(candidate_native)
        previous = _widen(
            cupy.nextafter(
                candidate_native, cupy.asarray(-cupy.inf, dtype=target_dtype)
            )
        )
        following = _widen(
            cupy.nextafter(candidate_native, cupy.asarray(cupy.inf, dtype=target_dtype))
        )
        offset = candidate - total
        virtual_negative_total = offset - candidate
        virtual_candidate = offset - virtual_negative_total
        offset_error = (candidate - virtual_candidate) + (
            -total - virtual_negative_total
        )
        lower_limit = cupy.nextafter(offset + (previous - candidate) * 0.5, cupy.inf)
        upper_limit = cupy.nextafter(offset + (following - candidate) * 0.5, -cupy.inf)
    certified = (
        valid
        & cupy.isfinite(candidate)
        & (offset_error == 0.0)
        & (lower_error > lower_limit)
        & (upper_error < upper_limit)
    )
    return candidate.reshape(kept_shape), certified.reshape(kept_shape)


def certified_float_sum(values: Any, axes: tuple[int, ...]) -> Any | None:
    """Return a correctly rounded native sum, or decline without guessing."""
    if not axes:
        return values
    with _errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        working = (
            _widen(values)
            if values.dtype.itemsize < cupy.dtype(cupy.float64).itemsize
            else values
        )
        direct = cupy.sum(working, axis=axes, keepdims=True, dtype=working.dtype)
        nan = cupy.any(cupy.isnan(values), axis=axes, keepdims=True)
        positive_infinity = cupy.any(cupy.isposinf(values), axis=axes, keepdims=True)
        negative_infinity = cupy.any(cupy.isneginf(values), axis=axes, keepdims=True)
        finite = cupy.all(cupy.isfinite(values), axis=axes, keepdims=True)
        absolute = cupy.abs(values)
        nonzero = cupy.isfinite(values) & (values != 0.0)
        if values.dtype.itemsize == 8:
            unsigned_type = cupy.uint64
            fraction_mask = cupy.uint64((1 << 52) - 1)
            hidden_bit = cupy.uint64(1 << 52)
            exponent_shift = cupy.uint64(52)
            exponent_mask = cupy.uint64(0x7FF)
            exponent_bias = 1023
            fraction_bits = 52
            subnormal_exponent = -1074
        else:
            unsigned_type = cupy.uint32
            fraction_mask = cupy.uint32((1 << 23) - 1)
            hidden_bit = cupy.uint32(1 << 23)
            exponent_shift = cupy.uint32(23)
            exponent_mask = cupy.uint32(0xFF)
            exponent_bias = 127
            fraction_bits = 23
            subnormal_exponent = -149
        bits = absolute.view(unsigned_type)
        exponent_field = (bits >> exponent_shift) & exponent_mask
        mantissa = bits & fraction_mask
        mantissa = cupy.where(exponent_field == 0, mantissa, mantissa | hidden_bit)
        low_bit = mantissa & ((~mantissa) + unsigned_type(1))
        trailing_zeros = cupy.where(nonzero, cupy.log2(low_bit).astype(cupy.int64), 0)
        base_exponent = cupy.where(
            exponent_field == 0,
            subnormal_exponent,
            exponent_field.astype(cupy.int64) - exponent_bias - fraction_bits,
        )
        value_exponent = base_exponent + trailing_zeros
        quantum_exponent = cupy.min(
            cupy.where(nonzero, value_exponent, 4096),
            axis=axes,
            keepdims=True,
        )
        safe_exponent = cupy.where(quantum_exponent == 4096, 0, quantum_exponent)
        quantum = cupy.ldexp(
            cupy.ones_like(safe_exponent, dtype=values.dtype), safe_exponent
        )
        units = cupy.where(nonzero, absolute / quantum, 0.0)
        total_units = cupy.sum(units, axis=axes, keepdims=True, dtype=values.dtype)
        precision = cupy.finfo(values.dtype).nmant + 1
        exact_lattice = (total_units <= float(2**precision)) & cupy.isfinite(
            total_units * quantum
        )
        minimum = cupy.min(values, axis=axes, keepdims=True)
        maximum = cupy.max(values, axis=axes, keepdims=True)
        same_sign_overflow = cupy.isinf(direct) & ((minimum >= 0.0) | (maximum <= 0.0))
    nonfinite = nan | positive_infinity | negative_infinity
    at_most_one_addition = math.prod((values.shape[axis] for axis in axes)) <= 2
    basic_finite = at_most_one_addition | exact_lattice | same_sign_overflow
    compensated, compensated_safe = _compensated_candidate(working, axes, values.dtype)
    finite_safe = basic_finite | compensated_safe
    certified = nonfinite | (finite & finite_safe)
    if not bool(cupy.all(certified)):
        return None
    finite_result = cupy.where(basic_finite, direct, compensated)
    both_infinities = positive_infinity & negative_infinity
    result = cupy.where(
        nan | both_infinities,
        cupy.nan,
        cupy.where(
            positive_infinity,
            cupy.inf,
            cupy.where(negative_infinity, -cupy.inf, finite_result),
        ),
    )
    return cupy.where(finite & (result == 0.0), 0.0, result)


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
    accumulator = cupy.ones(grouped.shape[0], dtype=cupy.uint64)
    overflow = cupy.zeros(grouped.shape[0], dtype=cupy.bool_)
    maximum = cupy.uint64(2**64 - 1)
    for index in range(grouped.shape[1]):
        factor = cupy.where(zero, cupy.uint64(1), magnitude[:, index])
        safe_factor = cupy.where(factor == 0, cupy.uint64(1), factor)
        overflow |= accumulator > maximum // safe_factor
        accumulator = cupy.where(overflow, cupy.uint64(0), accumulator * factor)
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
