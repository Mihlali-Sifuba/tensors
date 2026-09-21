# ReLU semantics

The forward numerical contract for `ts.relu`. It follows
[sign-semantics.md](sign-semantics.md), [abs-semantics.md](abs-semantics.md)
and [sqrt-semantics.md](sqrt-semantics.md) as the fourth operation migrated
off "whatever the Python backend happens to do" and onto a written
specification that every backend must meet exactly.

Scope is deliberately narrow: **forward `relu` only**. Differentiation — the
`relu_gradient` kernels, `execute_relu_gradient`, `ReLU.backward` and
`ReLU.backward_graph` — is outside this milestone and unchanged. In
particular this document says nothing about the derivative at zero or about
the gradient's NaN behaviour; both remain as they were and ungoverned.

## 1. The contract

ReLU is a unary elementwise rectification:

```
relu(x) = x   when x > 0
relu(x) = 0   otherwise
```

with NaN handled by §1.5 rather than by that rule.

ReLU is **class E**: every non-NaN result is exact. No tolerance and no ULP
budget applies. A conforming backend returns the specified value bit for bit.

### 1.1 Shape and dtype

- The output shape equals the input shape.
- The output dtype equals the input dtype. ReLU never promotes, never widens
  and never narrows, and the output dtype never depends on the input
  *values*.

Both hold for all seven public dtypes: `float64`, `float32`, `int64`,
`int32`, `int16`, `int8` and `uint8`.

### 1.2 Values

| input | result |
| --- | --- |
| finite and positive | itself, exactly |
| finite and negative | zero in the declared dtype |
| integer `0` | `0` |
| `+0.0` | `+0.0` |
| `-0.0` | `+0.0` |
| `+inf` | `+inf` |
| `-inf` | `+0.0` |
| `nan` (floating only) | `nan` |

`uint8` is unsigned, so every representable value is non-negative and ReLU is
the **identity** on it.

An integer operand is never routed through `float64`. `relu` of a large
`int64` — above `2**53`, where `float64` could not hold it — returns that
exact value.

### 1.3 Signed zero

Both `+0.0` and `-0.0` produce **canonical positive zero**: `-0.0` is not
greater than zero, so it takes the zero branch, and the zero the branch
produces is positive. The sign bit of the result is clear in both cases.

ReLU therefore agrees with [abs](abs-semantics.md#13-signed-zero) and
[sign](sign-semantics.md#13-signed-zero) and differs from
[sqrt](sqrt-semantics.md#13-zeros-and-why-sqrt-differs-from-abs-and-sign),
which keeps the sign. The three rules were each decided on their own
operation's terms and are not expected to match.

### 1.4 Subnormals

A **positive** subnormal is finite and greater than zero, so it is returned
**unchanged** and must never be flushed. This holds down to the smallest
`float32` subnormal, `1.401298464324817e-45`, and at `float64`.

A **negative** subnormal is not greater than zero, so it produces canonical
`+0.0` like any other negative value.

ReLU is the second operation, after [abs](abs-semantics.md#16-subnormals),
whose *result* can itself be subnormal — it returns the operand — so on CUDA
both the operand and the result cross the format boundary at risk. Measured
on this toolchain, `cupy.maximum` on a native binary32 array returns zero for
the smallest **and** the largest positive subnormal, and a CuPy conversion of
a binary64 result back down flushes again. The CUDA kernel therefore widens
through `ieee32.widen`, rectifies in binary64, and narrows through
`ieee32.narrow`. This is the same two-sided protection `abs` needs, and more
than `sqrt` needs, whose result is never subnormal.

### 1.5 NaN

A floating `nan` operand produces a `nan` result. This is a **classification
rule only**. The specification promises that the result is a NaN and nothing
more: no payload is promised, and no sign bit is promised. Tests must assert
`isnan` and must not assert a bit pattern, a payload or a sign.

NaN is the reason the rectification is not written as a comparison. A
comparison sends NaN to the zero branch and loses it; the maximum-based
expressions used by the array backends propagate it. Both were measured
before the implementation was chosen.

Integer dtypes have no NaN, so the rule does not apply to them.

### 1.6 No input is an error

Every value has a specified result. There is no domain error, no
unrepresentable input, and therefore **nothing to detect before executing**.

That distinguishes ReLU from [abs](abs-semantics.md#21-the-one-permitted-device-synchronisation),
whose signed-integer minimum requires a bounded device synchronisation.
Forward ReLU performs no host inspection of operand values, no reduction and
no device-to-host boolean read of any kind.

## 2. Where it executes

`relu` follows [Execution requirements](backends.md#execution-requirements) in
full. Backend selection is an execution requirement, not a preference:

- Under explicit `python`, `numpy` or `cuda` selection, `relu` executes on
  that backend's kernel — for every dtype, integers included.
- **No workload-size policy applies.** A one-element `relu` runs on the
  selected backend exactly as a million-element `relu` does.
- **There is no Python-reference fallback.** A selected backend that cannot
  produce a conforming result raises `BackendOperationUnsupportedError`
  rather than quietly running the work elsewhere.
- The result is resident on the selected backend. Residency is validated on
  the operand before execution and on the result after it.
- No tensor value is transferred to Python for numerical evaluation.

## 3. Division of responsibility

| layer | owns |
| --- | --- |
| `ReLU.forward` | Tensor semantics: the output dtype and output shape, both unchanged from the operand, and adopting the returned storage |
| `execute_relu` | Backend semantics: the selection, operand residency, lowering, kernel invocation, result residency |
| backend `relu` kernels | Numerical work only, on prepared native values, returning backend-native `Storage` |

A kernel never receives a `Tensor` and never inspects Tensor metadata. The
lowering branches are written out per backend — ReLU is unary, so there is no
scalar or broadcasting case, and this milestone deliberately does not
introduce a generic unary-dispatch layer, a preparation helper or a registry.

## 4. Fusion

`relu` is in the CUDA fusion planner's fusible set, and a fused `relu` must
produce exactly what an eager `relu` produces — the same requirement
[autodiff.md](autodiff.md) places on fused execution generally.

Measured at the fusion workload size, the planner does fuse it and the fused
kernel **executes** rather than declining, so the comparison is not answered
by a fallback. Eager and fused agree, and both satisfy this document, across
ordinary values, signed zeros, subnormals, infinities and NaN. The fusion
implementation was not modified by this milestone.

## 5. Conformance

The specification is testable from §1.2 alone. Expectations in the test suite
are literal, derived from this document, and are never obtained by running
the Python, NumPy or CUDA backend and recording what it returned. No backend
is a reference implementation for `relu`, and the fused path is checked
against the specification rather than only against the eager path, so a
defect shared by both cannot pass.
