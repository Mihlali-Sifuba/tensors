# Abs semantics

The forward numerical contract for `ts.abs`. It follows
[sign-semantics.md](sign-semantics.md) as the second operation migrated off
"whatever the Python backend happens to do" and onto a written specification
that every backend must meet exactly.

Scope is deliberately narrow: **forward `abs` only**. Differentiation — the
`abs_gradient` path, `Abs.backward` and `Abs.backward_graph` — is outside this
milestone and unchanged. Nothing here states a derivative rule.

## 1. The contract

`abs` is a unary elementwise magnitude. It is **exact**: every result below is
a specified value, not an approximation. No tolerance applies, and no backend
is the oracle for another.

### 1.1 Shape and dtype

- The output shape equals the input shape.
- The output dtype equals the input dtype. `abs` never promotes, never widens
  and never narrows, and the output dtype never depends on the input
  *values*.

Both hold for all seven public dtypes: `float64`, `float32`, `int64`,
`int32`, `int16`, `int8` and `uint8`.

### 1.2 Values

| input | result |
| --- | --- |
| finite and negative | its magnitude, exactly |
| finite and positive | itself |
| `+0.0` | `+0.0` (canonical positive zero) |
| `-0.0` | `+0.0` (canonical positive zero) |
| integer `0` | `0` |
| `+inf` | `+inf` |
| `-inf` | `+inf` |
| `nan` (floating only) | `nan` |
| a signed integer dtype's least value | **raises `OverflowError`** (§1.7) |

`uint8` is unsigned, so `abs` is the identity on it and can never fail.

### 1.3 Signed zero

Both `+0.0` and `-0.0` produce **canonical positive zero**. The sign bit of
the result is clear in both cases, so `copysign(1.0, abs(-0.0))` is `+1.0`.

### 1.4 NaN

A floating `nan` operand produces a `nan` result. This is a **classification
rule only**. The specification promises that the result is a NaN and nothing
more: no payload is promised, and no sign bit is promised. Tests must assert
`isnan` and must not assert a bit pattern, a payload or a sign.

Integer dtypes have no NaN, so the rule does not apply to them.

### 1.5 Infinities

`abs(+inf)` and `abs(-inf)` are both `+inf`. Neither is an error.

### 1.6 Subnormals

A subnormal operand is a finite nonzero operand, and `abs` returns its
magnitude exactly:

- a positive `float32` subnormal, down to the smallest at
  `1.401298464324817e-45`, is returned unchanged;
- a negative `float32` subnormal returns that same positive subnormal.

The same holds at `float64`. A subnormal is never flushed to zero, on the way
in or on the way out.

That last clause is load-bearing on CUDA, and `abs` is more exposed here than
`sign` is. Binary32 crosses the format boundary **twice**, and both crossings
flush on this toolchain: `cupy.abs` reads a binary32 operand with
flush-to-zero in force, and a CuPy conversion of the binary64 magnitude back
down to binary32 flushes again. `sign` only ever produces `-1`, `0`, `+1` or
NaN, so its narrowing was exact whatever the operand was; `abs` returns the
operand's *magnitude*, which can itself be subnormal. The CUDA kernel
therefore widens through `ieee32.widen` **and** narrows through
`ieee32.narrow`, rather than letting `CudaStorage` convert the binary64
result.

### 1.7 The signed integer minimum

A signed integer dtype's least value has a magnitude one greater than its
greatest value, so that magnitude is not representable:

| dtype | least value | magnitude | representable? |
| --- | --- | --- | --- |
| `int8` | `-128` | `128` | no (max `127`) |
| `int16` | `-32768` | `32768` | no (max `32767`) |
| `int32` | `-2147483648` | `2147483648` | no |
| `int64` | `-9223372036854775808` | `9223372036854775808` | no |

`abs` **raises `OverflowError`** when any element equals its dtype's least
value, with exactly this message:

```
abs(-128) is not representable in int8
```

— that is, `f"abs({minimum}) is not representable in {dtype.name}"`, naming
the dtype's least value rather than the element's position. The error is
identical on every backend, and it is raised before any kernel runs.

**This is not wraparound.** Returning `-128` — which is what `numpy.abs` and
`cupy.abs` both do natively, and what a typed buffer would produce by
wrapping — is a negative absolute value, and therefore an incorrect result
rather than a defined consequence of finite width. `docs/arithmetic-semantics.md`
§4.2 confines wraparound to `+ - * /` and `**`; `abs` is not arithmetic under
that contract and does not inherit it.

Four alternatives were considered and rejected: widening the result to a
larger signed dtype (`int64` has none, and it would make the output dtype
differ from the input); converting to floating point (`int64` values above
`2**53` are not representable in `float64`, which would make an exact
operation inexact); saturating to the dtype's maximum (silently wrong);
and wrapping (as above).

The rule is value-dependent, but only in *whether it raises*. The output
dtype and shape stay static, so nothing downstream has to inspect values to
know the result type.

## 2. Where it executes

`abs` follows [Execution requirements](backends.md#execution-requirements) in
full. Backend selection is an execution requirement, not a preference:

- Under explicit `python`, `numpy` or `cuda` selection, `abs` executes on
  that backend's kernel — for every dtype, integers included.
- **No workload-size policy applies.** A one-element `abs` runs on the
  selected backend exactly as a million-element `abs` does.
- **There is no Python-reference fallback.** A selected backend that cannot
  produce a conforming result raises `BackendOperationUnsupportedError`
  rather than quietly running the work elsewhere.
- The result is resident on the selected backend. Residency is validated on
  the operand before execution and on the result after it.

### 2.1 The one permitted device synchronisation

Detecting the §1.7 input on CUDA requires knowing whether any element equals
the dtype's least value. The comparison and the reduction both run **on the
device**; only the single resulting boolean is read back, so that the
specified `OverflowError` can be raised immediately rather than surfacing
later as a wrong value.

This is a deliberate, bounded exception to the rule in
[backends.md](backends.md#storage-residency-and-transfers) that ordinary
execution must not synchronise. It is confined to signed integer dtypes —
floating dtypes and `uint8` never reach it — and **no operand value is
transferred to Python**. It is not a fallback.

## 3. Division of responsibility

| layer | owns |
| --- | --- |
| `Abs.forward` | Tensor semantics: the output dtype and output shape, both unchanged from the operand, and adopting the returned storage |
| `execute_abs` | Backend semantics: the selection, operand residency, lowering, the §1.7 precheck, kernel invocation, result residency |
| backend `abs` kernels | Numerical work only, on prepared native values already known to be representable, returning backend-native `Storage` |

A kernel never receives a `Tensor` and never inspects Tensor metadata. A
kernel also never discovers the §1.7 rule for itself: by the time it runs,
every value it holds has a representable magnitude. The lowering and the
detection are written out per branch in `execute_abs` rather than shared —
`abs` is unary, so there is no scalar or broadcasting case, and this
milestone deliberately does not introduce a generic unary-dispatch layer.

## 4. Fusion

`abs` is in the CUDA fusion planner's fusible set, and a fused `abs` must
produce exactly what an eager `abs` produces — the same requirement
[autodiff.md](autodiff.md) places on fused execution generally. This is
covered by an exact eager-versus-fused test over ordinary, subnormal, signed
zero, infinite and NaN operands. The fused generator was already correct at
the time this specification was written and was not modified.

## 5. Conformance

The specification is testable from the tables alone. Expectations in the test
suite are literal, derived from this document, and are never obtained by
running the Python, NumPy or CUDA backend and recording what it returned. No
backend is a reference implementation for `abs`.
