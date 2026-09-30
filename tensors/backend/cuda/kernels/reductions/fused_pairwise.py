"""Fused indexed product reduction on CUDA, with the package's exact tree.

For every row ``r`` of an ``(rows, width)`` problem this computes

    pairwise_sum( fl(left[li[r, 0]] * right[ri[r, 0]]), ...,
                  fl(left[li[r, n-1]] * right[ri[r, n-1]]) )

with the tree of docs/summation-semantics.md, the non-finite classification
of its section 9 and the canonical zero of its section 10, bit for bit what
``pairwise_float_sum(left[li] * right[ri], (1,))`` produces. It never
materialises the gathered factors, the products or the tree levels in global
memory, and it does not know what the indices mean: callers such as the
convolution VJPs prepare them.

**Arithmetic.** Binary32 products and sums use the explicit round-to-nearest
PTX instructions ``mul.rn.f32`` and ``add.rn.f32``, as
:mod:`tensors.backend.cuda.kernels.arithmetic.ieee32` does: they keep
subnormals, which CuPy's generated float32 code flushes on this toolchain,
and cannot be contracted into a fused multiply-add. Binary64 uses
``__dmul_rn`` and ``__dadd_rn``, which are likewise never contracted. The
kernels are compiled with ``--fmad=false`` as a second guard.

**The tree in one block.** A block loads up to ``CHUNK`` consecutive terms of
one row into shared memory and runs the specified levels on them: at a level
holding ``c`` values, value ``j`` of the next level is ``s[2j] + s[2j+1]``
for ``j < c // 2``, and when ``c`` is odd the last value is carried
unchanged. Two shared buffers alternate between levels, with a barrier
between levels, so every addition reads only completed values of the
previous level; the arithmetic dependency graph is the specified tree
whatever the thread layout. Each term's non-finite class — 1 for NaN, 2 for
``+inf``, 4 for ``-inf`` — travels through the same levels, combined with
bitwise OR, which is exact.

**Rows wider than one block.** ``CHUNK`` is a power of two, ``2**t``. A row of
``n > CHUNK`` terms is split into chunks starting at multiples of ``CHUNK``,
and stage 1 reduces each chunk completely. This is exactly the first ``t``
levels of the row's tree:

- at level ``k < t`` a full chunk holds ``2**(t-k)`` values, an even number,
  so its pairs never cross a chunk boundary and it has no carry;
- the last chunk may be partial, but it is the last one, so any odd carry it
  makes is the row's odd carry at that level, which is what the row's tree
  does;
- a chunk of ``m <= CHUNK`` values is reduced to one value within ``t``
  levels, since ``ceil(m / 2**t) = 1``; a partial chunk that finishes early
  carries its single value unchanged through the remaining levels, as the
  row's tree carries an odd final value.

After stage 1 the ``ceil(n / CHUNK)`` chunk results, in order, are therefore
the values of level ``t`` of the row's tree. Later stages apply the same
argument to that sequence until one value remains. No level is reordered and
no chunk total is combined out of order.

**Finalisation.** In the stage that leaves one value per row, the class bits
decide the result — any NaN, or both infinities, gives NaN; otherwise an
infinity gives itself — and a zero result, detected from its bit pattern so a
subnormal is never mistaken for zero, is written as ``+0.0``.

Shared memory per block is two ``CHUNK``-element value buffers and two
``CHUNK``-element ``int32`` flag buffers: 16 KiB for float32 and 24 KiB for
float64. A stage's only global workspace is its per-chunk values and flags.

This module is internal to the CUDA backend. No facade re-exports it.
"""

from __future__ import annotations

from typing import Any

import cupy

#: Terms per block. A power of two, so chunk boundaries align with the tree.
CHUNK = 1024
_THREADS = 256

_SOURCE = r"""
#define CHUNK 1024

__device__ __forceinline__ float t_mul(float a, float b) {
    float r; asm volatile("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ float t_add(float a, float b) {
    float r; asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b)); return r;
}
__device__ __forceinline__ double t_mul(double a, double b) { return __dmul_rn(a, b); }
__device__ __forceinline__ double t_add(double a, double b) { return __dadd_rn(a, b); }

__device__ __forceinline__ int t_class(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int magnitude = bits & 0x7fffffffu;
    if (magnitude > 0x7f800000u) return 1;
    if (magnitude == 0x7f800000u) return (bits >> 31) ? 4 : 2;
    return 0;
}
__device__ __forceinline__ int t_class(double v) {
    unsigned long long bits = (unsigned long long)__double_as_longlong(v);
    unsigned long long magnitude = bits & 0x7fffffffffffffffull;
    if (magnitude > 0x7ff0000000000000ull) return 1;
    if (magnitude == 0x7ff0000000000000ull) return (bits >> 63) ? 4 : 2;
    return 0;
}

__device__ __forceinline__ float t_final(float tree, int flags) {
    if ((flags & 1) || ((flags & 6) == 6)) return __uint_as_float(0x7fc00000u);
    if (flags & 2) return __uint_as_float(0x7f800000u);
    if (flags & 4) return __uint_as_float(0xff800000u);
    if ((__float_as_uint(tree) & 0x7fffffffu) == 0u) return __uint_as_float(0u);
    return tree;
}
__device__ __forceinline__ double t_final(double tree, int flags) {
    if ((flags & 1) || ((flags & 6) == 6)) return __longlong_as_double(0x7ff8000000000000ll);
    if (flags & 2) return __longlong_as_double(0x7ff0000000000000ll);
    if (flags & 4) return __longlong_as_double((long long)0xfff0000000000000ull);
    if ((((unsigned long long)__double_as_longlong(tree)) & 0x7fffffffffffffffull) == 0ull)
        return __longlong_as_double(0ll);
    return tree;
}

// Reduce the `count` values in `a` (flags `fa`) with the specified tree,
// alternating with `b` / `fb`. Returns the buffer holding the result.
template <typename T>
__device__ __forceinline__ int t_tree(T* a, T* b, int* fa, int* fb, int count) {
    int in_a = 1;
    while (count > 1) {
        T* src = in_a ? a : b;
        T* dst = in_a ? b : a;
        int* fsrc = in_a ? fa : fb;
        int* fdst = in_a ? fb : fa;
        int half = count >> 1;
        for (int j = threadIdx.x; j < half; j += blockDim.x) {
            dst[j] = t_add(src[2 * j], src[2 * j + 1]);
            fdst[j] = fsrc[2 * j] | fsrc[2 * j + 1];
        }
        if ((count & 1) && threadIdx.x == 0) {
            dst[half] = src[count - 1];
            fdst[half] = fsrc[count - 1];
        }
        __syncthreads();
        in_a = !in_a;
        count = half + (count & 1);
    }
    return in_a;
}

template <typename T>
__device__ void t_products(
    const T* left, const T* right,
    const long long* left_index, const long long* right_index,
    long long rows, long long width, long long chunks,
    T* out, int* out_flags, int final)
{
    __shared__ T a[CHUNK];
    __shared__ T b[CHUNK];
    __shared__ int fa[CHUNK];
    __shared__ int fb[CHUNK];
    long long block = blockIdx.x;
    long long row = block / chunks;
    long long chunk = block - row * chunks;
    long long first = chunk * CHUNK;
    int count = (int)min((long long)CHUNK, width - first);
    const long long* li = left_index + row * width + first;
    const long long* ri = right_index + row * width + first;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        T p = t_mul(left[li[i]], right[ri[i]]);
        a[i] = p;
        fa[i] = t_class(p);
    }
    __syncthreads();
    int in_a = t_tree<T>(a, b, fa, fb, count);
    if (threadIdx.x == 0) {
        T tree = in_a ? a[0] : b[0];
        int flags = in_a ? fa[0] : fb[0];
        if (final) {
            out[row] = t_final(tree, flags);
        } else {
            out[block] = tree;
            out_flags[block] = flags;
        }
    }
}

template <typename T>
__device__ void t_values(
    const T* values, const int* flags_in,
    long long rows, long long width, long long chunks,
    T* out, int* out_flags, int final)
{
    __shared__ T a[CHUNK];
    __shared__ T b[CHUNK];
    __shared__ int fa[CHUNK];
    __shared__ int fb[CHUNK];
    long long block = blockIdx.x;
    long long row = block / chunks;
    long long chunk = block - row * chunks;
    long long first = chunk * CHUNK;
    int count = (int)min((long long)CHUNK, width - first);
    const T* v = values + row * width + first;
    const int* f = flags_in + row * width + first;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        a[i] = v[i];
        fa[i] = f[i];
    }
    __syncthreads();
    int in_a = t_tree<T>(a, b, fa, fb, count);
    if (threadIdx.x == 0) {
        T tree = in_a ? a[0] : b[0];
        int flags = in_a ? fa[0] : fb[0];
        if (final) {
            out[row] = t_final(tree, flags);
        } else {
            out[block] = tree;
            out_flags[block] = flags;
        }
    }
}

extern "C" __global__ void products_f32(
    const float* left, const float* right, const long long* li, const long long* ri,
    long long rows, long long width, long long chunks, float* out, int* out_flags, int final)
{ t_products<float>(left, right, li, ri, rows, width, chunks, out, out_flags, final); }

extern "C" __global__ void products_f64(
    const double* left, const double* right, const long long* li, const long long* ri,
    long long rows, long long width, long long chunks, double* out, int* out_flags, int final)
{ t_products<double>(left, right, li, ri, rows, width, chunks, out, out_flags, final); }

extern "C" __global__ void values_f32(
    const float* values, const int* flags, long long rows, long long width, long long chunks,
    float* out, int* out_flags, int final)
{ t_values<float>(values, flags, rows, width, chunks, out, out_flags, final); }

extern "C" __global__ void values_f64(
    const double* values, const int* flags, long long rows, long long width, long long chunks,
    double* out, int* out_flags, int final)
{ t_values<double>(values, flags, rows, width, chunks, out, out_flags, final); }
"""

_MODULE: Any = None


def _kernel(name: str) -> Any:
    global _MODULE
    if _MODULE is None:
        _MODULE = cupy.RawModule(code=_SOURCE, options=("--fmad=false", "-std=c++11"))
    return _MODULE.get_function(name)


def _suffix(dtype: Any) -> str:
    if dtype == cupy.float32:
        return "f32"
    if dtype == cupy.float64:
        return "f64"
    raise TypeError(f"fused pairwise reduction supports float32 and float64, not {dtype}")


def pairwise_indexed_product_sum(
    left_values: Any,
    right_values: Any,
    left_indices: Any,
    right_indices: Any,
) -> Any:
    """Reduce ``fl(left[li] * right[ri])`` row by row with the package's tree.

    ``left_values`` and ``right_values`` are flat device arrays of one
    floating dtype. ``left_indices`` and ``right_indices`` are ``(rows,
    width)`` ``int64`` device arrays naming each row's terms in order. The
    result is a ``(rows,)`` array of the values' dtype.
    """
    dtype = left_values.dtype
    if right_values.dtype != dtype:
        raise TypeError("both factors must have the same dtype")
    suffix = _suffix(dtype)
    left_values = cupy.ascontiguousarray(left_values).reshape(-1)
    right_values = cupy.ascontiguousarray(right_values).reshape(-1)
    left_indices = cupy.ascontiguousarray(left_indices, dtype=cupy.int64)
    right_indices = cupy.ascontiguousarray(right_indices, dtype=cupy.int64)
    rows, width = (int(left_indices.shape[0]), int(left_indices.shape[1]))
    result = cupy.empty((rows,), dtype=dtype)
    if rows == 0:
        return result
    if width == 0:
        result.fill(0)
        return result
    chunks = -(-width // CHUNK)
    final = chunks == 1
    stage_values = result if final else cupy.empty((rows * chunks,), dtype=dtype)
    stage_flags = (
        cupy.empty((1,), dtype=cupy.int32)
        if final
        else cupy.empty((rows * chunks,), dtype=cupy.int32)
    )
    _kernel(f"products_{suffix}")(
        (rows * chunks,),
        (_THREADS,),
        (
            left_values,
            right_values,
            left_indices,
            right_indices,
            cupy.int64(rows),
            cupy.int64(width),
            cupy.int64(chunks),
            stage_values,
            stage_flags,
            cupy.int32(1 if final else 0),
        ),
    )
    width = chunks
    while not final:
        chunks = -(-width // CHUNK)
        final = chunks == 1
        next_values = result if final else cupy.empty((rows * chunks,), dtype=dtype)
        next_flags = (
            cupy.empty((1,), dtype=cupy.int32)
            if final
            else cupy.empty((rows * chunks,), dtype=cupy.int32)
        )
        _kernel(f"values_{suffix}")(
            (rows * chunks,),
            (_THREADS,),
            (
                stage_values,
                stage_flags,
                cupy.int64(rows),
                cupy.int64(width),
                cupy.int64(chunks),
                next_values,
                next_flags,
                cupy.int32(1 if final else 0),
            ),
        )
        stage_values, stage_flags, width = next_values, next_flags, chunks
    return result


__all__ = ["CHUNK", "pairwise_indexed_product_sum"]
