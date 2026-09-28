# Sqrt semantics

The forward numerical contract for `ts.sqrt`. It follows
[sign-semantics.md](sign-semantics.md) and [abs-semantics.md](abs-semantics.md)
as the third operation migrated off "whatever the Python backend happens to
do" and onto a written specification that every backend must meet exactly.

Scope, stated exactly. This document governs:

- the **forward** result (sections 1 to 5);
- the **first-order VJP**, `Sqrt.backward` and `execute_sqrt_gradient`
  (section 6);
- the **recorded first-order VJP**, `Sqrt.backward` and the
  internal `SqrtVJP` operation, which must agree with section 6 in every
  respect (section 7);
- the **higher-order regions and boundaries** named in section 8, and only
  those.

It does not govern any other operation's gradient, and it does not claim
that autodiff through `sqrt` is specified beyond the regions section 8
names.

## 1. The contract

`sqrt` is a unary elementwise square root. Unlike `sign` and `abs` it is not
an exact operation on every input — the true square root is usually
irrational — so the contract is stated as **correct rounding** rather than as
a value table alone.

### 1.1 Shape and dtype

- The output shape equals the input shape.
- A floating operand keeps its format: `float32 → float32`,
  `float64 → float64`.
- An integer operand is **converted to `float64` first**, and the output
  dtype is `float64`. This holds for `int64`, `int32`, `int16`, `int8` and
  `uint8`.

The output dtype depends on the input *dtype*, never on the input *values*.

The integer rule is a **conversion rule, not a precision promise**. An
`int64` operand is rounded to the nearest `float64` before the root is taken,
so for values above `2**53` the root is the root of the converted operand,
not of the mathematical integer. Every backend performs that conversion
explicitly and identically.

### 1.2 Correct rounding

For a finite positive floating operand, the result is the **correctly
rounded** square root in the output dtype, under round-to-nearest, ties to
even. There is no tolerance: a conforming backend returns the one
representable value nearest the true root.

This is the operation's accuracy class, and it is testable without a backend
oracle — see §5.

### 1.3 Zeros, and why sqrt differs from abs and sign

| input | result |
| --- | --- |
| `+0.0` | `+0.0` |
| `-0.0` | `-0.0` |

**`sqrt` preserves the sign of zero.** `sqrt(-0.0)` is `-0.0`, not `+0.0`.
This is deliberate and is the opposite of [abs](abs-semantics.md#13-signed-zero)
and [sign](sign-semantics.md#13-signed-zero), which both canonicalise to
`+0.0`. IEEE 754 specifies `sqrt(-0)` as `-0`, and `sqrt` is a rounding
operation on the real line rather than a magnitude or a classification, so it
has no reason to discard the sign. Do not "fix" this to match the other two.

### 1.4 Negative operands do not raise

| input | result |
| --- | --- |
| finite and negative | `nan` |
| `-inf` | `nan` |

A negative operand is a **value with a specified result**, not a domain
error. `sqrt(-1.0)` is `nan`; it does not raise `ValueError`.

This changes previous behaviour, in which all three backends raised
`ValueError("sqrt is only defined for non-negative values")`. Three
consequences follow, and they are the point of the change:

- Nothing needs to inspect operand values before executing. There is no
  `any(...)` reduction in the dispatcher or in a kernel, and in particular
  **no CUDA host synchronisation** — the opposite of the bounded, deliberate
  one that [abs](abs-semantics.md#21-the-one-permitted-device-synchronisation)
  requires for its unrepresentable input.
- An integer operand follows the same rule after conversion, so a negative
  integer gives `float64` `nan`.
- The result stays elementwise: one bad element no longer denies the whole
  tensor a result.

**This establishes a non-trapping rule for `sqrt` alone.** It does not settle
the package-wide trapping policy, and it says nothing about `log`, the
inverse trigonometric functions, or any other operation that currently
raises on its domain. Those remain open.

### 1.5 Infinity

`sqrt(+inf)` is `+inf`. `sqrt(-inf)` is `nan` under §1.4.

### 1.6 NaN

A floating `nan` operand produces a `nan` result. This is a **classification
rule only**. The specification promises that the result is a NaN and nothing
more: no payload is promised, and no sign bit is promised. Tests must assert
`isnan` and must not assert a bit pattern, a payload or a sign.

### 1.7 Subnormals

A positive subnormal operand is a finite positive operand and gets the
correctly rounded root under §1.2. It is never flushed to zero. The smallest
`float32` subnormal, `1.401298464324817e-45`, has root
`3.743392066509216e-23`.

Note the asymmetry with the operand side: **`sqrt` can never produce a
subnormal result in `float32`**. A subnormal result would need an operand
below `2**-252`, which is far beneath the smallest `float32` subnormal at
`2**-149`. So only the *input* side of the format boundary is at risk, unlike
[abs](abs-semantics.md#16-subnormals), where the result itself can be
subnormal.

That asymmetry decides the CUDA implementation. `cupy.sqrt` reads a binary32
operand with flush-to-zero in force and reports the root of a subnormal as
zero, so the operand is widened through the PTX conversion in
`ieee32.py` first and the root is taken in binary64. Narrowing that binary64
root back down then yields the **correctly rounded binary32 result** — the
conversion itself still rounds; what it does not do is disagree with
rounding the true root directly. No dedicated binary32 PTX square-root
instruction is therefore needed: binary64 carries 53 significand bits, and
square root needs only `2p + 2 = 50` of them for those two roundings to
agree. This was not assumed — it was measured against the independent
reference of §5 over adversarial operands chosen so that their true roots sit
as close as possible to a `float32` rounding boundary.

## 2. Where it executes

`sqrt` follows [Execution requirements](backends.md#execution-requirements) in
full. Backend selection is an execution requirement, not a preference:

- Under explicit `python`, `numpy` or `cuda` selection, `sqrt` executes on
  that backend's kernel — for every dtype, integers included.
- **No workload-size policy applies.** A one-element `sqrt` runs on the
  selected backend exactly as a million-element `sqrt` does.
- **There is no Python-reference fallback.** A selected backend that cannot
  produce a conforming result raises `BackendOperationUnsupportedError`
  rather than quietly running the work elsewhere.
- The result is resident on the selected backend. Residency is validated on
  the operand before execution and on the result after it.

## 3. Division of responsibility

| layer | owns |
| --- | --- |
| `Sqrt.forward` | Tensor semantics: the output dtype (§1.1, including the integer rule) and output shape, and adopting the returned storage |
| `execute_sqrt` | Backend semantics: the selection, operand residency, lowering, kernel invocation, result residency |
| backend `sqrt` kernels | Numerical work only, on prepared native values: the explicit integer conversion, the root, and the format handling their provider needs |

A kernel never receives a `Tensor` and never inspects Tensor metadata. The
lowering hands each kernel the operand in the Tensor's own dtype, and the
conversion to `float64` for an integer operand is written out in each kernel
rather than hidden in a cast at the boundary. The lowering branches are
written out per backend — `sqrt` is unary, so there is no scalar or
broadcasting case, and this milestone deliberately does not introduce a
generic unary-dispatch layer.

## 4. Fusion

`sqrt` is in the CUDA fusion planner's fusible set, and fused execution must
observably agree with eager execution under this contract.

It does, by two different routes, and the distinction is worth recording. For
operands the fused kernel can handle, it runs and produces the specified
results. For an operand that is negative, the fused kernel still carries the
older domain guard, so the fused attempt **declines** and the caller re-runs
the steps eagerly — which now produces the specified `nan`. The observable
result is the contract's in both cases, so the fusion implementation was not
modified by this milestone.

That decline is a wasted launch rather than a wrong answer, and removing the
now-obsolete guard is a separate, fusion-scoped change.

## 5. Conformance

Expectations in the test suite are derived from this document, and the
correct-rounding requirement of §1.2 is checked against an **independent
reference**, never against another backend.

That reference computes the correctly rounded root from exact integer
arithmetic: the operand is taken as the exact rational it is, an integer
square root brackets the true root between two rationals whose gap can be
made arbitrarily small, and each endpoint is rounded directly into the target
format by `round_fraction` from the arithmetic reference — the same validated
routine `docs/arithmetic-semantics.md` §12.6.5 uses, which rounds once from a
rational rather than through an intermediate format. The result is accepted
only when both endpoints round to the same representable value; otherwise the
bracket is tightened and the evaluation repeats. No Python, NumPy or CUDA
square root takes part in producing an expected value.

## 6. The first-order VJP

For an upstream gradient `g` and the primal input `x`, the VJP is
`g / (2 * sqrt(x))`. Unlike [sign](sign-semantics.md#6-the-first-order-vjp)
and [abs](abs-semantics.md#6-the-first-order-vjp) this is arithmetic rather
than routing, so the contract has to say **in what order and in what
precision** it is evaluated.

### 6.1 The evaluation order is part of the contract

Elementwise, in the **declared dtype**:

```
root        = correctly_rounded_sqrt(x)
denominator = correctly_rounded_multiply(2, root)
result      = correctly_rounded_divide(g, denominator)
```

Each of those three steps rounds in the declared dtype. A `float32` VJP is
**not** evaluated in binary64 and narrowed once at the end: that would give
a different, more accurate answer than the specified sequence, and a
specification that does not say which is meant is not a specification.

`correctly_rounded_sqrt` is [section 1.2](#12-correct-rounding) of this
document; the multiplication and the division are the correctly rounded
operations of [arithmetic-semantics.md](arithmetic-semantics.md).

### 6.2 Exceptional primals

| `x` | result |
| --- | --- |
| `> 0` | the three steps above |
| `+0.0` or `-0.0` | **raises** `ValueError: sqrt derivative is undefined at zero` |
| negative | `nan` |
| `nan` | `nan` |

A negative primal is a **value**, consistent with
[section 1.4](#14-negative-operands-do-not-raise): the root is NaN and NaN
carries through the remaining two steps, so no backend inspects operand
values to find it and no host synchronisation is needed for it.

Only the zero primal raises, and both signed zeros do.

### 6.3 Subnormals

A subnormal primal is nonzero, so it takes the ordinary path rather than
raising — the same distinction [sign's VJP](sign-semantics.md#63-the-one-permitted-device-synchronisation)
has to make, and for the same reason: on CUDA a binary32 subnormal compares
equal to zero natively, so the zero test runs on the widened operand.

A subnormal **upstream gradient** reaches the division intact, and the
result of the division may itself be subnormal. On CUDA the doubling and
the division therefore use `ieee32.apply`, the same round-to-nearest PTX
instructions the arithmetic contract uses; ordinary CuPy binary32
operations would flush at both points.

### 6.4 Shape, dtype and residency

- `grad.shape` must equal `value.shape`, and `grad.dtype` must equal
  `value.dtype`; both are checked in `Sqrt.backward` and raise `ValueError`.
- The result has `value`'s shape and dtype.
- Only `float32` and `float64` reach here, because only those may require
  gradients. No integer differentiation is defined.
- Both operands and the result are validated resident on the selected
  backend. No workload threshold applies and there is no Python-reference
  fallback; a declining kernel raises `BackendOperationUnsupportedError`.

## 7. The graph-built VJP

`Sqrt.backward` records the VJP as a graph vertex through the internal
`SqrtVJP` operation. For the same operands,

```python
ts.grad(output, value, create_graph=False)
ts.grad(output, value, create_graph=True).data
```

agree in value, NaN classification, signed zero, dtype, shape,
selected-backend residency and domain errors.

The previous implementation read the materialised host values to decide the
zero domain and then built `grad / (2 * sqrt(x))` from public operations.
The expression was differentiable, but the **domain check was not part of
it**: it was taken once, at graph-build time, from the values present then.
A graph built over positive values and replayed onto a zero primal would not
have raised. Recording `SqrtVJP` carries the check into the replay.

`SqrtVJP` is internal. No facade re-exports it, and it is not public API.

## 8. Higher-order regions

| region | rule |
| --- | --- |
| `x > 0` | the VJP is genuinely differentiable, and higher derivatives follow from the graph |
| `x` is zero | the first VJP raises, so no higher derivative exists to take |
| negative `x` and NaN `x` | the first VJP is NaN and NaN propagates; nothing falls back to Python |

The two partial derivatives of the first VJP are kept apart. With respect to
the **upstream gradient** the VJP is linear, so that partial is
`1 / (2 * sqrt(x))` scaled by the outer gradient — this operation applied
again. With respect to the **primal** it is `-g / (4 * x * sqrt(x))`, built
from the governed operations so that any further derivative follows from the
graph rather than from another hand-written rule.

One implementation note that is easy to get wrong: the minus sign in that
second partial is carried by the scalar rather than by negating the
numerator. `negate` has not been migrated, so it still applies a
workload threshold and answers a small tensor with Python storage, which the
following division then rejects as a residency mismatch. A higher-order rule
may only be built from operations whose execution is already governed.

### 8.1 A limit on third and higher order, measured here

The **second** derivative is available on every backend. **Third** order and
beyond currently work on the Python backend only, and the cause is outside
this operation: differentiating that far reaches `multiply`'s own VJP, which
routes through `sum_products_to_shape` with an operand that a
still-unmigrated operation answered with Python storage, and the residency
check rejects it.

This was measured, not inferred, and it is not a `sqrt` defect: a plain
``v * v * v`` third derivative fails identically on NumPy with no `sqrt`
anywhere in the expression. It is the same class of defect as the
creation-kernel declines — a strict dispatcher paired with an unmigrated
kernel — and it resolves when the arithmetic VJP path's remaining operations
are migrated. Until then this document claims third order for the Python
backend only.
