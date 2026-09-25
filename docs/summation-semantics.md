# Summation semantics

The numerical contract for floating reductions, and the algorithm intended
to implement it.

**Status: specification and validated algorithm; exact accumulation is not yet
implemented.** The contract below is decided. The algorithm below is prototyped
and checked against an exact oracle. NumPy and CUDA currently use the sound
native certification boundary described in §7: they return a value only when
correct rounding is established, and otherwise report the operation as
unsupported. The cost measured in §5 and the product decision in §6 still
block a complete native implementation.

## 1. The contract

For `float32` and `float64` reductions:

> **Sum the represented finite input values exactly, then round once to the
> declared output dtype, round-to-nearest, ties-to-even.**

Consequences, each of which is a requirement:

- Cancellation survives even when an intermediate floating sum would
  overflow. `[1e308, 1e308, -1e308, -1e308]` sums to `0.0`, not NaN.
- A representable result never becomes an infinity or a zero through
  intermediate range loss.
- A `float32` result rounds **directly** to `float32`. Rounding through
  `float64` and narrowing is not assumed equivalent and is not permitted as
  the implementation.
- Subnormal operands and subnormal results are preserved.
- Permuting the inputs, or changing how the reduction is partitioned, does
  not change the result.
- A final value beyond the format's range rounds to the correctly signed
  infinity.
- Any NaN input gives NaN.
- Both `+inf` and `-inf` present in one group gives NaN.
- `+inf` without `-inf` or NaN gives `+inf`; the mirror case gives `-inf`.
- An exactly zero finite sum gives canonical `+0.0` — including a sum that
  cancels to zero, and a group containing only negative zeros.
- An empty reduction group gives `+0.0`.
- NaN payload and NaN sign are unspecified.

An empty reduction *group* is distinct from a zero-sized *output* and from
an axis selection that reduces nothing; the existing axis, shape, `keepdims`
and dtype-selection behaviour is unchanged.

**Scope.** This governs floating summation only. Integer reduction
behaviour is unchanged and is not given a new policy here.

This is a specification-owned contract. Python, NumPy and CUDA are three
implementations of it. None is the reference for the others.

## 2. Why the obvious algorithms do not qualify

**Native `sum`** is order-dependent and loses cancellation. Measured
against the stable Python reference over 400 ordinary random reductions,
the NumPy kernel differed in **101** of them — the disagreement is routine,
not exotic.

**Pairwise summation** does not satisfy the contract and must not be
described as though it does. For `[1e16, 1.0, -1e16, 1.0] * 8` the exact
sum is `16.0`, and an adjacent-pair reduction tree returns `0.0`.

**Compensated (Kahan/Neumaier) summation** carries one correction term. It
improves accuracy but is neither exact nor overflow-tolerant: the contract
requires a finite answer where an intermediate would overflow, and a
compensated accumulator overflows with the sum.

## 3. The algorithm: exact fixed-point accumulation

### 3.1 Accumulator representation

Every finite binary64 is exactly `s · m · 2**q` with `m` a 53-bit integer
and `q` in `[-1074, 971]`. So **every finite binary64 is an integer
multiple of `2**-1074`**, and `|x| < 2**1024`.

Working in units of `2**-1074` therefore turns a floating sum into an
**integer** sum. The accumulator is a signed fixed-point integer with that
unit, held as `int64` limbs of `B = 24` bits.

For binary32 the same argument gives units of `2**-149` and `|x| < 2**128`.

### 3.2 How cancellation is preserved

It is not preserved — it is *exact*. No rounding happens during
accumulation, so `1e16 + 1.0 - 1e16 + 1.0` loses nothing: the four integers
cancel in the fixed-point accumulator and the `1.0` terms remain. Range
loss cannot occur either, because no intermediate is ever a float.

### 3.3 Capacity against reduction length

An N-term binary64 sum needs `1074 + 1024 + ceil(log2 N)` bits, so
`L = ceil(2098 / 24) + 3 = 91` limbs covers any reduction up to
`2**(3·24)` terms — far beyond any representable tensor.

Each element contributes to at most 6 limb slots: the 53-bit mantissa is
split into three ≤24-bit pieces, each piece shifted by at most 23 bits
stays below `2**47`, and each lands across two adjacent limbs. A limb
therefore grows by less than `2**47` per element, so an `int64` limb absorbs
more than `2**15` elements before needing a carry, and a carry
normalisation pass restores headroom.

### 3.4 Combining partial accumulators

Two accumulators combine by limb-wise integer addition — associative,
commutative and exact. Carries are propagated once, at the end. This is
what makes a partitioned or parallel reduction give the same answer as a
serial one.

### 3.5 Order independence

Integer addition is associative and commutative, and no step rounds. The
result depends only on the multiset of inputs, so permutation and partition
cannot change it. This is an argument, not a sampling result.

### 3.6 Final rounding

From the exact fixed-point integer:

1. Locate the most significant set bit to get the result exponent.
2. Take the leading 53 bits (24 for binary32) as the candidate significand.
3. Form the round bit and the sticky bit from everything below it.
4. Apply round-to-nearest, ties-to-even.
5. If the exponent is below the format's minimum normal, re-round at the
   subnormal quantum instead — which is why the accumulator's unit is the
   subnormal quantum, so this needs no separate path.
6. If the rounded value exceeds the format's maximum, return the signed
   infinity.
7. An accumulator of exactly zero returns `+0.0`.

Rounding happens **once**, from the exact value, directly into the
destination format. That is what makes the binary32 result correct without
a double-rounding argument.

Non-finite inputs are classified before accumulation: any NaN gives NaN;
both infinities give NaN; one infinity gives that infinity.

## 4. Validation of the algorithm

The accumulator was prototyped in NumPy and compared against an exact
Python-integer oracle built from `float.as_integer_ratio`. It reproduced
the exact sum on cancellation (`[1e16, 1, -1e16, 1] * 8`), intermediate
overflow (`[1e308, 1e308, -1e308, -1e308]`), subnormals, mixed magnitudes
spanning `2**-200` to `2**200`, and random values.

That validates the *accumulation*. The rounding path of §3.6 is specified
but not yet prototyped, and it is where the remaining risk sits.

## 5. Measured cost

Exact accumulation against `numpy.sum`, float64, one reduction:

| elements | `numpy.sum` | exact | ratio |
| --- | --- | --- | --- |
| 1,000 | 14.3 µs | 272.6 µs | 19× |
| 100,000 | 30.8 µs | 14.3 ms | 464× |
| 1,000,000 | 599.6 µs | 139.1 ms | 232× |

The cost is dominated by the scatter into limbs, not by arithmetic.

**Memory.** A grouped reduction needs one accumulator per output element:
`groups × 91 × 8` bytes. Reducing `(1e6, 10)` along the last axis therefore
needs about **728 MB** of accumulator for an 80 MB input.

## 6. The decision this forces

Applying §3 unconditionally would make every floating reduction — and so
every broadcast gradient reduction — 19× to 464× slower, with the memory
cost above. That is a product decision, not an implementation detail:

1. **Accept the cost.** Simplest and unconditionally correct.
2. **Fast path with an exact fallback.** Take a native sum plus a *sound*
   certificate that it is already correctly rounded, and fall back to §3
   only when the certificate fails. Both paths native, so the contract and
   residency both hold, and ordinary data pays almost nothing. The
   certificate has to be provably sound; an unsound one silently breaks the
   contract.
3. **Restrict the contract's scope**, for example to gradient reductions
   only, leaving `ts.sum` as it is.

Option 2 is the recommendation. It is also materially more work than
option 1 and needs its own soundness argument.

## 7. What the package does today

Until the exact accumulator is implemented:

- The Python backend uses its stable reference summation, including the exact
  ratio recovery needed when a temporary binary64 total would overflow.
- NumPy and CUDA perform the reduction on their own native values only when a
  sound certificate proves the provider result conforming. The certificate
  accepts IEEE non-finite classifications, empty and zero groups, a single
  correctly rounded addition, finite groups whose exact binary lattice sum fits
  both the dtype significand and its intermediate range. The certificate uses
  only a bounded number of provider-native array operations; it never iterates
  over reduction elements in Python.
- A finite group that is not certified returns `None`; strict dispatch turns
  that into `BackendOperationUnsupportedError`. There is no Python fallback.
- `reduce_sum`, `sum_to_shape`, and gradient accumulation through `Sum` share
  this boundary, so an uncertified broadcast gradient fails explicitly rather
  than returning an inaccurate derivative.

This preserves the numerical contract and selected-backend residency while
leaving full support for all finite groups to the exact accumulator described
above.
