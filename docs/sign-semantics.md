# Sign semantics

The forward numerical contract for `ts.sign`. This is the first operation
migrated off "whatever the Python backend happens to do" and onto a written
specification that every backend must meet exactly.

Scope, stated exactly. This document governs:

- the **forward** result (§1–§5);
- the **first-order VJP**, `Sign.backward` and `execute_sign_gradient` (§6);
- the **recorded first-order VJP**, `Sign.backward` and the
  internal `SignVJP` operation, which must agree with §6 in every respect
  (§7);
- the **higher-order regions and boundaries** named in §8, and only those.

It does not govern any other operation's gradient, and it does not claim
that autodiff through `sign` is specified beyond the regions §8 names.

## 1. The contract

`sign` is a unary elementwise classification. It reports which side of zero
each element lies on, and it is **exact**: every result below is a specified
value, not an approximation. No tolerance applies, and no backend is the
oracle for another.

### 1.1 Shape and dtype

- The output shape equals the input shape.
- The output dtype equals the input dtype. `sign` never promotes, never
  widens and never narrows.

Both hold for all seven public dtypes: `float64`, `float32`, `int64`,
`int32`, `int16`, `int8` and `uint8`.

### 1.2 Values

| input | result |
| --- | --- |
| finite and negative | `-1` |
| finite and positive | `+1` |
| `+0.0` | `+0.0` (canonical positive zero) |
| `-0.0` | `+0.0` (canonical positive zero) |
| integer `0` | `0` |
| `+inf` | `+1` |
| `-inf` | `-1` |
| `nan` (floating only) | `nan` |

Read in the declared dtype: for a floating dtype the results are `-1.0`,
`+0.0` and `+1.0`; for an integer dtype they are `-1`, `0` and `+1`.

### 1.3 Signed zero

Both `+0.0` and `-0.0` produce **canonical positive zero**. The sign bit of
the result is clear in both cases, so `copysign(1.0, sign(-0.0))` is `+1.0`.
`sign` does not preserve the operand's zero sign; it reports that the operand
is neither negative nor positive.

### 1.4 NaN

A floating `nan` operand produces a `nan` result. This is a **classification
rule only**. The specification promises that the result is a NaN and nothing
more: no payload is promised, and no sign bit is promised. Tests must assert
`isnan` and must not assert a bit pattern, a payload or a sign.

Integer dtypes have no NaN, so the rule does not apply to them.

### 1.5 Subnormals

A subnormal operand is a finite nonzero operand and is classified as one:

- a positive `float32` subnormal, down to the smallest at
  `1.401298464324817e-45`, produces `+1`;
- a negative `float32` subnormal produces `-1`.

The same holds for `float64` subnormals. A subnormal is never treated as
zero. This rule is what [§5.4 of the arithmetic
specification](arithmetic-semantics.md) requires of gradual underflow, stated
for this operation.

This one is load-bearing on CUDA. `cupy.sign` reads a binary32 operand with
flush-to-zero in force, so a subnormal arrives at the comparison as zero and
would be classified `0` rather than `±1`. The CUDA kernel therefore widens a
binary32 operand through the PTX conversion in
`tensors/backend/cuda/kernels/arithmetic/ieee32.py` before classifying.
Narrowing the result back is exact whatever the operand was, because `sign`
only ever produces `-1`, `0`, `+1` or NaN — none of which is subnormal.

### 1.6 Integers

Integer inputs follow the same `-1`, `0`, `+1` rule, in their own width.
`uint8` is unsigned, so it naturally yields only `0` and `+1`.

An integer operand is **never** routed through `float64`. Doing so would lose
`int64` values above `2**53`, and the classification is well defined in the
integer width itself.

## 2. Where it executes

`sign` follows [Execution requirements](backends.md#execution-requirements) in
full. Backend selection is an execution requirement, not a preference:

- Under explicit `python`, `numpy` or `cuda` selection, `sign` executes on
  that backend's kernel.
- **No workload-size policy applies.** A one-element `sign` runs on the
  selected backend exactly as a million-element `sign` does. There is no
  threshold below which the work is handed to the Python reference.
- **There is no Python-reference fallback.** A selected backend that cannot
  produce a conforming result raises `BackendOperationUnsupportedError`
  rather than quietly running the work elsewhere.
- The result is resident on the selected backend. Residency is validated on
  the operand before execution and on the result after it.

`"auto"` is not a third behaviour: it resolves to a concrete backend at
selection time.

## 3. Division of responsibility

| layer | owns |
| --- | --- |
| `Sign.forward` | Tensor semantics: the output dtype and output shape, both unchanged from the operand, and adopting the returned storage |
| `execute_sign` | Backend semantics: the selection, operand residency, lowering to the backend's native representation, kernel invocation, result residency |
| backend `sign` kernels | Numerical work only, on prepared native values, returning backend-native `Storage` |

A kernel never receives a `Tensor` and never inspects Tensor metadata. The
lowering is written out in each branch of `execute_sign` rather than shared:
`sign` is unary, so there is no scalar or broadcasting case, and this
milestone deliberately does not introduce a generic unary-dispatch layer.

## 4. Fusion

`sign` is in the CUDA fusion planner's fusible set, and a fused `sign` must
produce exactly what an eager `sign` produces — the same requirement
[autodiff.md](autodiff.md) places on fused execution generally. This is
covered by an exact eager-versus-fused test over ordinary, subnormal, signed
zero, infinite and NaN operands. The fused generator was already correct at
the time this specification was written and was not modified.

## 5. Conformance

The specification is testable from the table alone. Expectations in the test
suite are literal, derived from this document, and are never obtained by
running the Python, NumPy or CUDA backend and recording what it returned. No
backend is a reference implementation for `sign`.

## 6. The first-order VJP

For an upstream gradient `g` and the primal input `x`, elementwise:

| `x` | result |
| --- | --- |
| finite and nonzero | canonical `+0.0` |
| `+0.0` or `-0.0` | **raises** `ValueError: sign derivative is undefined at zero` |
| `nan` | `nan` |

`sign` is piecewise constant, so its derivative is zero wherever it exists.

### 6.1 The result does not depend on the upstream gradient

This is **routing, not multiplication**. The result is canonical `+0.0`
whatever `g` is:

| `g` at a finite nonzero `x` | result |
| --- | --- |
| negative | `+0.0`, never `-0.0` |
| `±inf` | `+0.0`, never `nan` |
| `nan` | `+0.0`, never `nan` |

Implementing this as `g * 0` is forbidden, and the reason is measured rather
than theoretical. Before this specification the Python backend routed and
returned a literal zero while NumPy and CUDA materialised a derivative and
multiplied, so the same call gave:

```
x = 2.0, g = -2.0   ->  python +0.0   numpy -0.0   cuda -0.0
x = 2.0, g = ±inf   ->  python +0.0   numpy  nan   cuda  nan
x = 2.0, g = nan    ->  python +0.0   numpy  nan   cuda  nan
```

The specification settles that disagreement in favour of routing, because a
piecewise-constant function's VJP carries no information from upstream.

### 6.2 Shape, dtype and residency

- `grad.shape` must equal `value.shape`, and `grad.dtype` must equal
  `value.dtype`; both are checked in `Sign.backward` and raise `ValueError`.
- The result has `value`'s shape and dtype.
- Only `float32` and `float64` reach here, because only those may require
  gradients. No integer differentiation is defined.
- Both operands and the result are validated resident on the selected
  backend. No workload threshold applies and there is no Python-reference
  fallback; a declining kernel raises `BackendOperationUnsupportedError`.

### 6.3 The one permitted device synchronisation

Enforcing the zero domain error on CUDA needs one on-device comparison and
reduction, whose single resulting boolean is read back so the `ValueError`
can be raised. No operand value is transferred.

That test runs on the **widened** operand, and the reason is measured: on
this toolchain a binary32 subnormal compares equal to zero natively, because
flush-to-zero reaches the comparison's operand. Testing natively would raise
"undefined at zero" for a subnormal, which §1.5 defines as an ordinary
nonzero value. The CUDA kernel widens through `ieee32.widen` before
comparing.

## 7. The graph-built VJP

`Sign.backward` records the VJP as a graph vertex through the internal
`SignVJP` operation. For the same operands,

```python
ts.grad(output, value, create_graph=False)
ts.grad(output, value, create_graph=True).data
```

agree in value, NaN classification, signed zero, dtype, shape,
selected-backend residency and domain errors.

the VJP reads no host values. It does not inspect `value.data._data`
and builds no Python list-comprehension mask. That matters for replay: a
compiled graph built over positive values and replayed over negative, zero or
subnormal values must answer for the values it is replayed with, including
raising the §6 zero error on replay. A frozen mask would answer for the
values the graph was built with.

`SignVJP` is internal. No facade re-exports it, and it is not public API.

## 8. Higher-order regions

| region | rule |
| --- | --- |
| finite nonzero `x` | the first VJP and **every** higher derivative are zero |
| `x` is zero | the first VJP raises, so no higher derivative exists to take |
| `x` is NaN | the first VJP is NaN, and so is its derivative — it is not silently turned into zero |

The derivative of the first VJP has the same form as the first VJP, so
`SignVJP` differentiates into itself. Nothing beyond these three regions is
specified.
