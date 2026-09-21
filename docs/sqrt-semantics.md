# Sqrt semantics

The forward numerical contract for `ts.sqrt`. It follows
[sign-semantics.md](sign-semantics.md) and [abs-semantics.md](abs-semantics.md)
as the third operation migrated off "whatever the Python backend happens to
do" and onto a written specification that every backend must meet exactly.

Scope is deliberately narrow: **forward `sqrt` only**. Differentiation — the
`sqrt_gradient` path, `execute_sqrt_gradient`, `Sqrt.backward` and
`Sqrt.backward_graph` — is outside this milestone and unchanged. Nothing here
states a derivative rule, and the derivative's own domain error at zero is
untouched.

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
