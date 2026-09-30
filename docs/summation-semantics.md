# Summation semantics

The numerical contract for multi-term floating accumulation: sums, the
reductions inside contractions, and gradient accumulation.

**Status: implemented on the Python, NumPy and CUDA backends.** Every backend
executes the algorithm below natively on its own storage. None refuses an
ordinary finite reduction, and none answers through another backend.

## 1. The contract

> **Floating summation uses a deterministic balanced pairwise reduction tree
> in the declared dtype.**

A floating sum is **not** defined as the exact real sum rounded once. It is
defined as a specified sequence of correctly rounded additions, and the
result of that sequence is the answer, including where it differs from the
exact sum.

Two things are deliberately *not* relaxed:

- **Elementary arithmetic is unchanged.** `a + b`, `a - b`, `a * b` and
  `a / b` remain correctly rounded in the declared dtype, as
  [arithmetic semantics](arithmetic-semantics.md) specifies. The pairwise
  tree is built out of exactly those correctly rounded additions.
- **Integer reductions are unchanged.** They are exact, and overflow of the
  declared width raises `OverflowError`.

Backend selection decides where a sum runs. It does not decide how the sum is
computed.

## 2. The algorithm

For one reduction group `x[0], x[1], ..., x[n-1]`, taken in the logical order
of §3:

1. **Round 1.** Add adjacent pairs: `s[0] = fl(x[0] + x[1])`,
   `s[1] = fl(x[2] + x[3])`, and so on. If the round has an odd number of
   values, the final unpaired value is carried into the next round unchanged.
2. **Later rounds.** Repeat the same adjacent-pair step on the previous
   round's values: `s'[j] = fl(s[2j] + s[2j+1])`, carrying an odd final value.
3. Continue until one value remains. That value is the tree result.
4. Apply the non-finite classification of §9 and the zero rule of §10.

`fl` is one correctly rounded addition in the declared dtype (§4). No
provider may choose another tree: not NumPy's `sum`, not CuPy's `sum`, not a
BLAS blocking, not a thread schedule.

For example, `[a, b, c, d, e]` reduces as `((a + b) + (c + d)) + e`, and
`[a, b, c]` as `(a + b) + c`.

## 3. Logical order

The input sequence of a group is determined by the tensor's **logical**
values, never its physical layout:

- Reduced axes are taken in increasing axis order, whatever order the caller
  named them in.
- The group is the reduced coordinates in row-major order over those axes.
- Strides, offset, tiling, kernel launch geometry and provider partitioning
  do not enter.

So `ts.sum(x, axis=(3, 1))` and `ts.sum(x, axis=(1, 3))` reduce the same
sequence, and a transposed view reduces in its logical order.

## 4. Dtype rounding

Every addition in the tree rounds to the declared reduction dtype
immediately:

- `float64`: each addition is one binary64 addition.
- `float32`: each addition is one binary32 addition. A `float32` group is
  never accumulated in binary64 and narrowed at the end, which would be a
  different algorithm with different results.

How each backend achieves that:

- **Python** computes each two-operand addition in binary64, where the exact
  sum of two binary32 values fits, and rounds that single result to binary32
  before it enters the next round.
- **NumPy** adds `float32` arrays in `float32`.
- **CUDA** adds `float32` arrays with the explicit `add.rn.f32` PTX instruction
  (`kernels/arithmetic/ieee32.py`). CuPy's generated `float32` code flushes
  subnormals on this toolchain, and the explicit instruction does not.

Dtype promotion outside summation is unchanged.

## 5. Determinism and cross-backend identity

For a fixed logical value sequence, shape, reduction axes and dtype, the
result is deterministic. Each step is a correctly rounded IEEE addition, so
every backend that implements the same tree produces the **same bits** for
every finite and infinite result. NaN payload and NaN sign are unspecified.

Tests compare backends bit for bit, and every backend is compared against an
independent reference implementation of this tree (`tests/_pairwise_oracle.py`),
not against another backend.

## 6. Error bound

For a finite group of `n` elements without intermediate overflow, let
`k = ceil(log2 n)` be the depth of the tree and `S = sum(x)` the exact real
sum. With the unit roundoff `u` (`2**-24` for `float32`, `2**-53` for
`float64`) and the smallest positive subnormal `η` of the dtype:

```text
|Ŝ - S|  <=  ((1 + u)**k - 1) * sum(|x_i|)  +  (n - 1) * (η / 2) * (1 + u)**k
```

**Justification.** Model each addition as
`fl(a + b) = (a + b)(1 + δ) + ε` with `|δ| <= u` and `|ε| <= η/2`. Unrolling
the tree:

- each input `x_i` reaches the root through at most `k` additions, so it is
  multiplied by at most `k` factors `(1 + δ)`;
- each of the `n - 1` additions contributes one `ε`, which is then multiplied
  by at most `k` further factors.

Bounding `|Π(1 + δ) - 1| <= (1 + u)**k - 1` for each input, and
`|ε · Π(1 + δ)| <= (η/2)(1 + u)**k` for each addition, gives the bound.

The second term is conservative. With gradual underflow, an addition whose
result is subnormal is exact, so `ε` is in fact zero for additions. It is kept
so that the bound needs no appeal to that theorem.

For small `k·u`, `(1 + u)**k - 1 <= k·u / (1 - k·u)`. The error therefore
grows with `log2 n` rather than with `n`, as sequential summation's does.

The bound applies only without intermediate overflow (§8). Tests check it
against an exact rational oracle for random signs, cancellation, mixed
magnitudes, subnormals, and odd and even lengths, in both dtypes.

## 7. Underflow and subnormals

Gradual underflow is required. Subnormal inputs, intermediate values and
results take part according to the declared dtype. No backend enables or
relies on flush-to-zero; §4 describes how CUDA avoids it for `float32`.

## 8. Intermediate overflow

**Breaking change.** The previous contract summed exactly and so survived
intermediate overflow. It required `[1e308, 1e308, -1e308, -1e308]` to sum to
`0.0`. Under this contract:

1. round 1 gives `fl(1e308 + 1e308) = +inf` and `fl(-1e308 + -1e308) = -inf`;
2. round 2 gives `+inf + -inf = NaN`.

`NaN` is the specified result. No exact recovery is attempted. The algorithm
stays fully specified when an intermediate overflows; only the error bound of
§6 does not apply.

## 9. Non-finite inputs

A group containing non-finite **inputs** is classified explicitly, whatever
the tree would give:

| inputs | result |
| --- | --- |
| any NaN | NaN |
| both `+inf` and `-inf` | NaN |
| `+inf`, no `-inf`, no NaN | `+inf` |
| `-inf`, no `+inf`, no NaN | `-inf` |

So `[-1e308, -1e308, +inf]` is `+inf`, even though its first round
overflows to `-inf`. The classification runs on the selected backend; CUDA
values are never copied to the host to classify them.

## 10. Zero and empty groups

- An empty group sums to `+0.0`.
- A result that compares equal to zero is returned as canonical `+0.0`. That
  includes exact cancellation and groups of only `-0.0`.

No backend exposes a `-0.0` sum.

## 11. Contractions

A contraction is a product then a sum:

```text
C[i, j] = pairwise_sum( fl(A[i, 0] * B[0, j]), ..., fl(A[i, m-1] * B[m-1, j]) )
```

Each product is formed once and rounded to the dtype, then the products are
reduced in increasing contraction-index order with the tree of §2. No fused
multiply-add replaces the rounded product. Provider matrix products (BLAS,
cuBLAS, `numpy.matmul`) are not used, because their blocking and FMA choose a
different summation.

This applies to:

- **`dot` and `matmul`**, including batched and broadcast products, over the
  contracted axis.
- **The `matmul` VJPs**, in two stages. First the contraction over the
  broadcast batch (the left VJP sums over columns, the right VJP over rows).
  Then any batch axes the operand was broadcast along are summed back to its
  shape with the tree.
- **The `outer` VJPs**: each is a contraction over the other vector's index.
- **Convolution.** An output element contracts its whole receptive field:
  the group's input channels in order, and within each channel the kernel
  offsets in row-major order. A tap that falls in the padding is an exact
  zero term, not skipped, so every backend's tree has the same shape. A bias
  is then one more rounded addition. The array backends still refuse a padded
  convolution whose kernel contains a non-finite weight, because `0 * inf`
  would otherwise be used as data.
- **Convolution VJPs**, each over a fixed term sequence, with an exact zero
  for a combination that contributes nothing:
  - an input gradient runs over the kernel offsets and, within each offset,
    over the group's output channels;
  - a kernel gradient runs over the batch and, within it, the output positions;
  - a bias gradient sums the upstream over the batch and the output positions.
- **`ProductSumToShape`** (the broadcast VJP of `*`): products in the dtype,
  then the tree over the broadcast axes.

An overflowing product is therefore an infinity in its group. For example,
`fl(1e308 * 2)` and `fl(1e308 * -2)` are `+inf` and `-inf`, so the dot
product `[1e308, 1e308] · [2, -2]` is NaN.

## 12. Gradients

There are no separate gradient summation semantics:

- Broadcast gradient reductions (`sum_to_shape`) use the tree over the
  stretched axes.
- Contributions to one Variable from several uses are stacked in the order
  the reverse pass produces them and summed with the tree.
- The contraction VJPs follow §11.

A backward pass never fails because a reduction group cannot be proved
exactly rounded.

## 13. Execution and residency

- The selected backend executes the tree, and its result stays in that
  backend's storage. There is no fallback to Python and no migration between
  backends.
- NumPy and CUDA run each tree level as one native array operation over every
  group at once. Only the `O(log n)` levels are iterated in Python; no
  per-element Python loop runs on those backends.
- CUDA reads nothing back to the host to compute or classify a sum.
- `BackendOperationUnsupportedError` remains the answer for an operation a
  backend genuinely does not implement (for example integer matrix products
  on the array backends). An ordinary finite floating sum is always supported.

## 14. How the tree differs from the exact sum

| inputs | pairwise tree | exact sum | why |
| --- | --- | --- | --- |
| `[0.1, 0.2, 0.3]` | `0.6000000000000001` | `0.6` (rounded) | `fl(fl(0.1 + 0.2) + 0.3)` |
| `[1e16, 1, -1e16, 1]` | `0.0` | `2.0` | `fl(1e16 + 1) = 1e16` and `fl(-1e16 + 1) = -1e16` |
| `[-1e16, 1, 1e16, 1, -2]` | `-2.0` | `0.0` | left-to-right summation gives `-1.0` |
| `[1e308, 1e308, -1e308, -1e308]` | `NaN` | `0.0` | intermediate overflow, §8 |
| `[1e308, 1e308, -1e308]` | `+inf` | `1e308` | `+inf` carried, then `+inf - 1e308` |

For `float32`, `[-1.0, 2**-24, -2**-24, -0.001, -0.001]` (inputs rounded to
binary32) reduces to `-1.002000093460083` with per-addition rounding. It
would be `-1.0019999742507935` if the group were accumulated in binary64 and
narrowed once.

## 15. Scope

This contract governs multi-term floating accumulation performed by the
summation machinery: `ts.sum`, `sum_to_shape`, gradient accumulation, and the
contractions of §11. Operations with their own specified numerics are not
redefined by it: `mean`, `variance`, `std`, `norm`, `logsumexp`, softmax and
log-softmax, the losses and the optimizer updates. They run natively on
ordinary data on every backend.

## Appendix A. Historical: exact accumulation (not implemented, not required)

The previous contract was *"sum the represented finite inputs exactly, then
round once to the destination dtype"*. To meet it without refusing, the NumPy
and CUDA backends would have needed an exact fixed-point accumulator: every
finite binary64 is an integer multiple of `2**-1074`, so a sum can be
accumulated exactly in about 91 integer limbs of 24 bits and rounded once at
the end.

A prototype matched an exact oracle on cancellation, intermediate overflow,
subnormals and mixed magnitudes. It measured 19× to 464× slower than
`numpy.sum` (1,000 to 1,000,000 elements), with an accumulator of
`groups × 91 × 8` bytes. Until then the array backends used a certificate that
accepted only the few groups it could prove exactly rounded, and refused
ordinary data such as `[0.1, 0.2, 0.3]`.

That design is kept here only as background for a possible future opt-in
exact mode. It is not implemented and it is not the package contract. The
full earlier text is in the Git history of this file.
