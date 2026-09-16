# Arithmetic semantics

## 1. Status and scope

> **Status: approved target contract, awaiting implementation.**
>
> This document specifies what MS-Tensors arithmetic **will** do. It is not a
> description of what the package does today. Sections marked *Current
> behaviour* record the present implementation so the difference is visible;
> everywhere else, the specified behaviour is the target. Do not read this
> document as a description of the installed package, and do not cite it as
> evidence that a behaviour is already implemented.
>
> [Section 10](#10-migration-and-compatibility) lists the known differences
> between the specification and the current implementation.

### In scope

Elementwise **addition**, **subtraction** and **multiplication** where both
operands have the **same** dtype, for the four scalar numeric dtypes:

| dtype | `DataType` name | Typecode | Width |
| --- | --- | --- | --- |
| 32-bit signed integer | `int32` | `i` | 4 bytes |
| 64-bit signed integer | `int64` | `q` | 8 bytes |
| IEEE 754 binary32 | `float32` | `f` | 4 bytes |
| IEEE 754 binary64 | `float64` | `d` | 8 bytes |

**Division** is specified separately in [section 7](#7-division), because it
does not share the result-dtype rule of the other three operations.

### Out of scope

This document does **not** specify, and no part of it may be read as
authorising:

- mixed-dtype arithmetic or a promotion table ([section 6](#6-result-dtypes-and-promotion));
- integer division semantics ([section 7](#7-division));
- the dtypes `int16`, `int8` and `uint8`, which `tensors.dtype` defines but
  which this contract does not cover;
- comparison, reduction, linear-algebra, transcendental or fused operations;
- tensor–scalar arithmetic, where the scalar has no declared dtype;
- broadcasting rules, which are a shape concern
  ([section 3.4](#34-broadcasting-and-storage));
- automatic differentiation.

Where a rule is absent, it is absent because it has not been decided. Absence
is not permission to inherit a backend's default. Every undecided point is
listed in [section 11](#11-open-specification-questions).

---

## 2. Architectural principles

### 2.1 The specification is the authority

> **The declared dtype and this specification define an operation's semantics.
> Python, NumPy and CUDA are implementations of that specification. No backend
> is the semantic authority.**

Performance may differ between backends. The specified results, result dtypes
and exceptional behaviour must not.

This inverts the arrangement the package was built with. Today the Python
backend's behaviour *is* the definition: the NumPy and CUDA kernels are
written to reproduce it, and decline to run whenever they cannot. The
consequences are visible in the implementation:

- `tensors/backend/numpy/conversion.py` converts every floating operand to
  `numpy.float64` regardless of the declared dtype, and computes integer
  arithmetic in `object` arrays of Python integers.
- `tensors/backend/cuda/conversion.py` refuses integer dtypes outright, so
  every integer operation falls back to the Python kernel.
- A `float32` result that would round to infinity causes the NumPy kernel to
  decline, because the Python reference does not produce that overflow.

Under this specification the direction of authority reverses. A backend is
correct when it matches this document, not when it matches Python.

### 2.2 Independent of Python and NumPy

The specification is deliberately independent of:

- **Python's built-in numerical types.** Python's `int` is arbitrary-precision
  and its `float` is always binary64. Neither is a model for a typed tensor
  library. A tensor declares a width, and the width is the contract.
- **NumPy's incidental behaviour.** Where this document and NumPy agree, it is
  because the same standard was applied, not because NumPy was consulted.
  Where NumPy's behaviour is an artefact of its own history, it does not bind
  this package.

### 2.3 Declared dtype is the contract

An operation's semantics are determined by the **declared** dtype of its
operands, not by the values they happen to hold, the backend selected, the
storage representation in use, or the size of the tensor.

In particular, an operation must not change its semantics because a tensor is
small enough to take a different execution path, or large enough to take an
accelerated one.

---

## 3. Dtypes and representable values

### 3.1 Integer dtypes

Both integer dtypes are **signed** and **two's complement**.

| dtype | Width \(w\) | Minimum | Maximum |
| --- | --- | --- | --- |
| `int32` | 32 | `-2147483648` (\(-2^{31}\)) | `2147483647` (\(2^{31}-1\)) |
| `int64` | 64 | `-9223372036854775808` (\(-2^{63}\)) | `9223372036854775807` (\(2^{63}-1\)) |

These match `_INTEGER_LIMITS` in `tensors/dtype.py`.

### 3.2 Floating dtypes

| dtype | Format | Significand | Max finite | Min normal | Min subnormal | Epsilon |
| --- | --- | --- | --- | --- | --- | --- |
| `float32` | IEEE 754 binary32 | 24 bits | `3.4028235e+38` | `1.1754944e-38` | `1e-45` | `1.1920929e-07` |
| `float64` | IEEE 754 binary64 | 53 bits | `1.7976931348623157e+308` | `2.2250738585072014e-308` | `5e-324` | `2.220446049250313e-16` |

Each format represents signed zeros, signed infinities and NaNs in addition to
the finite values above.

### 3.3 Values outside the range

An integer value outside a dtype's range is not representable in that dtype.
What happens to such a value is the subject of [section 4](#4-integer-arithmetic):
it is **wrapped**, not rejected and not promoted.

A finite real number whose magnitude exceeds a floating dtype's maximum is not
representable either. What happens to it is the subject of
[section 5](#5-floating-point-arithmetic): it **rounds to infinity**, and that
infinity is a result, not an error.

### 3.4 Broadcasting and storage

Broadcasting determines **which pairs of elements** an operation combines.
This specification determines **what each pair produces**. The two are
independent: broadcasting a `(3, 1)` tensor against a `(3, 4)` one does not
change the arithmetic applied to any resulting pair, and no rule in this
document depends on operand shape.

Storage representation is likewise independent of semantics. A `float32`
tensor may be held in a Python `array('f')`, a `numpy.ndarray` of
`numpy.float32`, or a device-resident `cupy.ndarray`. All three must yield the
same values for the same operation. A backend may choose any internal
representation that produces the specified result, but the **declared dtype of
the result must be stored as that dtype**: a `float32` result is a `float32`
tensor, not a `float64` tensor carrying a `float32` label.

---

## 4. Integer arithmetic

### 4.1 Decided rules

For `int32` and `int64`, where **both operands have that same dtype**:

1. **Addition, subtraction and multiplication preserve the input dtype.**
   `int32 + int32 → int32`; `int64 * int64 → int64`.
2. **Arithmetic is fixed-width and two's complement, with wraparound.**
3. **Overflow does not raise.** For these three operations there is no
   overflow error, no promotion to a wider dtype, and no promotion to a
   floating dtype.
4. **Results are identical across every supported backend**, bit for bit.

### 4.2 The wraparound rule

Let \(r\) be the **mathematical** result of the operation — the exact value
that addition, subtraction or multiplication of the two operand values
produces over the integers, computed without any width limit and before any
wrapping is applied. For a signed integer dtype of width \(w\), the stored
result is

$$
\operatorname{wrap}_w(r) \;=\; \bigl((r + 2^{\,w-1}) \bmod 2^{\,w}\bigr) - 2^{\,w-1}
$$

where \(\bmod\) is the non-negative remainder (the result of
\(x \bmod 2^w\) lies in \([0,\, 2^w)\) for every integer \(x\), including
negative \(x\)).

Equivalently: \(\operatorname{wrap}_w(r)\) is the unique value in
\([-2^{w-1},\, 2^{w-1}-1]\) congruent to \(r\) modulo \(2^w\).

When \(r\) is already representable, \(\operatorname{wrap}_w(r) = r\), so the
rule describes ordinary arithmetic as well as overflow.

> \(r\) is a definitional device, not an implementation requirement. It
> specifies *which* value must be stored. It does not require an
> implementation to compute \(r\) exactly and then reduce it — see
> [section 4.4](#44-implementation-freedom).

### 4.3 Worked examples

Every value below was checked against the formula and against native
fixed-width arithmetic; the two agree in all cases.

| dtype | Expression | Mathematical \(r\) | Stored result |
| --- | --- | --- | --- |
| `int32` | `2147483647 + 1` | `2147483648` | `-2147483648` |
| `int32` | `-2147483648 - 1` | `-2147483649` | `2147483647` |
| `int32` | `1000000 * 1000` | `1000000000` | `1000000000` |
| `int32` | `65536 * 65536` | `4294967296` | `0` |
| `int32` | `2147483647 * 2147483647` | `4611686014132420609` | `1` |
| `int32` | `-2147483648 * -1` | `2147483648` | `-2147483648` |
| `int64` | `9223372036854775807 + 1` | `9223372036854775808` | `-9223372036854775808` |
| `int64` | `-9223372036854775808 - 1` | `-9223372036854775809` | `9223372036854775807` |
| `int64` | `4294967296 * 4294967296` | `18446744073709551616` | `0` |

Reading the first row through the formula, with \(w = 32\) and
\(r = 2147483648\):

$$
\operatorname{wrap}_{32}(2147483648)
= \bigl((2147483648 + 2^{31}) \bmod 2^{32}\bigr) - 2^{31}
= 0 - 2147483648
= -2147483648
$$

Note the last `int32` row: `-2147483648 * -1` wraps to itself. The negation of
the minimum value is not representable, so \(\operatorname{wrap}\) returns the
minimum again. This is a consequence of the rule, not an exception to it.

### 4.4 Implementation freedom

An implementation may compute the result by any means that produces the
specified value. It may use native fixed-width machine arithmetic that wraps
in hardware, compute in a wider type and reduce, or mask the low \(w\) bits.

**Arbitrary-precision arithmetic is not required and must not be assumed to be
required.** The current NumPy path computes integer arithmetic in `object`
arrays of Python integers in order to reproduce Python's arbitrary-precision
intermediates. Under this specification that is unnecessary: native `int32`
and `int64` arithmetic already wraps exactly as the rule requires, which is
the cheapest correct implementation on every backend.

---

## 5. Floating-point arithmetic

### 5.1 Decided rules

1. **Same-dtype arithmetic preserves the dtype.** `float32 op float32 →
   float32`; `float64 op float64 → float64`, for addition, subtraction,
   multiplication and division.
2. **Operations follow the declared precision.** An implementation must not
   unconditionally widen `float32` operands to `float64`.
3. **An implementation must not compute in `float64` and narrow to `float32`
   merely to imitate Python's built-in `float`.** Section 5.5 explains why
   this matters, and precisely where it does and does not change results.
4. **Floating-point overflow is a result, not a fallback.** An operation whose
   correctly-rounded result exceeds the dtype's finite range produces a signed
   infinity. It must not raise, and must not cause a backend to decline and
   defer to the Python kernel.

The numerical foundation is **IEEE 754**: binary32 for `float32`, binary64 for
`float64`.

### 5.2 Rounding

Addition, subtraction, multiplication and division are **correctly rounded**:
the result is the representable value nearest the exact mathematical result,
computed as if with unbounded range and precision and then rounded once.

The rounding mode is **round-to-nearest, ties-to-even**, the IEEE 754 default.
No other rounding mode is specified, and an implementation must not select one.

Correct rounding is what makes cross-backend agreement attainable: for these
four operations the result is *uniquely determined* by the two operand values,
the dtype and the rounding mode. There is no latitude for a conforming
implementation to differ. This is not true of transcendental functions, which
IEEE 754 does not require to be correctly rounded, and which this document
does not cover.

Example, at the rounding boundary:

| dtype | Expression | Result | Why |
| --- | --- | --- | --- |
| `float32` | `1.0 + 2**-24` | `1.0` | exactly half an ulp; ties-to-even selects the even significand |
| `float32` | `1.0 + 2**-23` | `1.0000001` | one ulp above `1.0`, exactly representable |

### 5.3 Infinities, NaNs and signed zero

| Case | `float32` | `float64` |
| --- | --- | --- |
| Overflow: `3.0e38 + 3.0e38` | `inf` | (no overflow at this magnitude) |
| Overflow: `max * 2` | `inf` | `inf` |
| `inf + 1.0` | `inf` | `inf` |
| `inf - inf` | `nan` | `nan` |
| `0.0 * inf` | `nan` | `nan` |
| `nan + 1.0` | `nan` | `nan` |
| `nan == nan` | `False` | `False` |

- **Infinities** are produced by overflow and propagate through arithmetic.
  They are values, not errors.
- **NaN** propagates: any arithmetic operation with a NaN operand produces
  NaN. NaN compares unequal to everything, including itself.
- **NaN payloads and the sign bit of a NaN are not specified.** A conformance
  test must treat all NaNs as equivalent and must not compare NaN bit
  patterns.
- **Signed zero** is preserved as IEEE 754 defines it: `0.0 + -0.0` is `+0.0`
  under round-to-nearest, `-0.0 + -0.0` is `-0.0`, and `-0.0 == 0.0` is
  `True`. The sign of a zero is observable through division
  (`1.0 / -0.0` is `-inf`) and must not be discarded.

The current implementation already propagates infinities and NaNs correctly
and identically on all three backends for addition and subtraction; this was
checked directly. What changes under this specification is overflow, which
must stop triggering a fallback.

### 5.4 Subnormals — partially open

IEEE 754 specifies **gradual underflow**: a result too small to be normal is
represented as a subnormal rather than flushed to zero. For `float32`:

| Expression | IEEE 754 result |
| --- | --- |
| `min_normal / 2` | `5.877472e-39` (subnormal) |
| `min_subnormal / 2` | `0.0` (underflows to zero) |

**Whether every backend must implement gradual underflow is an open
question.** It is measured, not assumed — see
[section 9.3](#93-what-cross-backend-equality-is-achievable) and
[open question O4](#o4-subnormal-handling-on-cuda). The CUDA backend
as currently configured flushes subnormals to zero, and that is the *only*
source of cross-backend disagreement found.

### 5.5 Why declared precision matters

This is the subtlest part of the contract, and the naive argument for it is
wrong. It is worth stating precisely, because an implementer who believes the
wrong version will draw the wrong conclusions about what must be fixed.

**A single widened operation is not observably different.** If two `float32`
operands are widened to binary64, one operation is performed, and the result
is rounded once to binary32, the result is **bit-identical** to native
binary32 arithmetic. This is the classical safe-double-rounding property:
double rounding through a wider format is innocuous when the wider format has
at least \(2p + 2\) bits, and binary64's 53 bits exceed the
\(2 \times 24 + 2 = 50\) that binary32 requires. This was verified directly:
across 200,000 random `float32` pairs per operation, native binary32 and
widen-compute-narrow agreed bit for bit on **every** pair, for addition,
subtraction, multiplication and division.

So widening is not wrong because it corrupts a single operation. It is wrong
for three other reasons:

**(a) The intermediate range is different.** binary64 has a far wider exponent
range, so an intermediate that overflows binary32 survives in binary64:

```text
a = 1e30, b = 1e20, c = 1e25   (all float32)

per-operation float32 : (a * b) / c  ->  inf
float64 throughout    : (a * b) / c  ->  1.0000001e+25
```

A `float32` computation that overflows must overflow. Carrying the
intermediate in binary64 silently rescues it, which is a different
computation, not a more accurate one.

**(b) Precision is retained across operations.** The safe-rounding property
covers *one* operation. As soon as a widened intermediate is fed to the next
operation without being rounded back to binary32, the results diverge. Across
200,000 random `float32` triples, `(a * b) + c` computed per-operation in
binary32 differed from the same expression evaluated entirely in binary64 in
**25%** of cases:

```text
a = 38.24823, b = 280.5834, c = -0.4536956

per-operation float32 : 10731.364   (bits 0x4627ad75)
float64 throughout    : 10731.365   (bits 0x4627ad76)
```

This matters wherever an evaluation strategy spans more than one operation —
fusion, in particular — and it means "we round the final result to float32"
is not a sufficient defence.

**(c) It costs what it was meant to save.** Widening every operand allocates
and converts a buffer of twice the width, and then converts back. On the
accelerated backends this is the dominant cost of a small operation, and it
buys nothing, because per operation the result is identical.

The requirement is therefore stated in terms of what is *observable*: a
`float32` operation must produce the correctly-rounded binary32 result, must
overflow when binary32 overflows, and must produce a `float32` result stored
as `float32`. An implementation that achieves this by widening a single
operation is conforming but wasteful; one that widens across an expression is
not conforming.

---

## 6. Result dtypes and promotion

### 6.1 Decided: same-dtype operands

| Left | Right | Operation | Result dtype |
| --- | --- | --- | --- |
| `int32` | `int32` | `+` `-` `*` | `int32` |
| `int64` | `int64` | `+` `-` `*` | `int64` |
| `float32` | `float32` | `+` `-` `*` | `float32` |
| `float64` | `float64` | `+` `-` `*` | `float64` |

Division is not in this table; see [section 7](#7-division).

### 6.2 Undecided: mixed dtypes

> **Mixed-dtype arithmetic is not specified by this document.**

`tensors/dtype.py` contains a `result_dtype` function that today produces a
result dtype for mixed operands. Its existence is **not** a specification. It
was written to serve the current implementation, it has not been reviewed
against this contract, and it must not be treated as the decided rule.

**Promotion rules must be defined before any mixed-dtype implementation is
changed.** An implementation task acting on this document must:

- implement the same-dtype rules in [section 6.1](#61-decided-same-dtype-operands);
- leave mixed-dtype behaviour exactly as it is;
- not derive a promotion rule from NumPy, from Python, or from the existing
  `result_dtype` function.

Questions that a promotion specification will have to answer are listed as
[O1](#o1-mixed-dtype-promotion).

---

## 7. Division

Division is separated from the other three operations because its result dtype
does not follow theirs, and because its exceptional behaviour is unresolved.

### 7.1 Decided: floating division

For operands of the same floating dtype:

| Left | Right | Result dtype |
| --- | --- | --- |
| `float32` | `float32` | `float32` |
| `float64` | `float64` | `float64` |

Division is correctly rounded under round-to-nearest ties-to-even, exactly as
[section 5.2](#52-rounding) specifies for the other operations.

### 7.2 Undecided: division by zero

> **The behaviour of division by zero is an open specification question.**

IEEE 754 defines `x / 0.0` for finite non-zero `x` as a signed infinity, and
`0.0 / 0.0` as NaN, raising the *divide-by-zero* and *invalid* flags
respectively — flags, not exceptions.

**MS-Tensors does not currently do this.** Division by zero raises
`ZeroDivisionError`, and this is a deliberate, package-level decision rather
than a backend artefact: the guard appears in the operation layer
(`tensors/operations/arithmetic/divide.py`) and is mirrored in all three
backend kernels. It is consistent across backends today.

This document does **not** decide to change it. Adopting IEEE semantics here
would be a separate, deliberate change to a documented behaviour that users
may depend on, and it was not part of the decision this specification records.
See [O2](#o2-division-by-zero).

### 7.3 Undecided: integer division

> **Integer division semantics are an open specification question.**

Nothing in this document should be read as deciding that `/` on two integer
tensors preserves an integer dtype, that it produces a floating dtype, or that
it follows Python's true-division or floor-division rules.

Recorded for reference, not as specification: the current implementation
produces `float64` from `int32 / int32` and from `int64 / int64`.

An implementation task must not change integer division on the strength of
this document. See [O3](#o3-integer-division).

---

## 8. Backend conformance

### 8.1 Requirements

1. **Python, NumPy and CUDA implement the same contract.** All three are
   implementations; none is the definition.
2. **The Python backend is not the semantic reference.** Its being the
   simplest and most readable implementation makes it a useful cross-check,
   not an authority. Where the Python kernel and this document disagree, the
   Python kernel is wrong.
3. **A backend must not silently change dtype or arithmetic semantics.** It
   must not return a result of a different dtype than specified, and must not
   store a result in a wider representation than its declared dtype.
4. **A backend must not fall back to Python merely because native arithmetic
   differs from Python's built-in semantics.** Native fixed-width integer
   wraparound and native binary32 arithmetic are *correct* under this
   contract. They are the reason the fallbacks exist today, and they are not
   grounds for a fallback under this specification.
5. **Native kernels should be used wherever they can satisfy the contract.**

### 8.2 Conformance is not coverage

Semantic consistency and operation coverage are different questions, and this
document only settles the first.

Requiring that every backend produce the same result **does not** assert that
every operation is natively implemented on every backend. It is not the case
today and this document does not claim otherwise: integer arithmetic on CUDA
is currently rejected outright, and small workloads are routed to the Python
kernel by the workload policy described in
[Numerical backends](backends.md).

### 8.3 Legitimate fallback

An implementation may fall back to another backend's kernel when an operation
is **genuinely unsupported** there — the library is unavailable, the dtype has
no device equivalent, or no correct native implementation exists.

A fallback must:

- **preserve the contract.** A fallback is an execution choice. It must
  produce exactly the specified result, dtype and exceptional behaviour.
- **be explicit and observable.** It must be visible through diagnostics or
  benchmarking rather than inferred from a timing anomaly. The benchmark
  suite already records where a case executes; a fallback that cannot be seen
  cannot be reviewed.

A fallback is **not** legitimate when it exists to reproduce Python's
arithmetic semantics. That is the situation this specification removes.

---

## 9. Conformance testing requirements

This section describes the tests a future implementation must add. **No such
tests exist yet**, and nothing here has been executed.

### 9.1 What every case must assert

For each operation, dtype and backend, a conformance test must check:

1. the **value** of the result;
2. the **dtype** of the result, by identity (`assertIs(result.dtype, ts.int32)`);
3. the **storage width**, so a `float32` result is not held as `float64`;
4. **exceptional behaviour** — that overflow does not raise for the cases in
   [section 4](#4-integer-arithmetic) and produces infinity for those in
   [section 5](#5-floating-point-arithmetic).

### 9.2 Required boundary values

| Category | Values |
| --- | --- |
| `int32` boundaries | `-2147483648`, `-1`, `0`, `1`, `2147483647` |
| `int64` boundaries | `-9223372036854775808`, `-1`, `0`, `1`, `9223372036854775807` |
| Integer overflow | `max + 1`, `min - 1`, `min * -1`, `max * max`, `65536 * 65536` (`int32`) |
| `float32` magnitudes | `0.0`, `-0.0`, `min_subnormal`, `min_normal`, `1.0`, `max`, `inf`, `-inf`, `nan` |
| `float64` magnitudes | the same set at binary64 |
| Rounding | `1.0 + 2**-24`, `1.0 + 2**-23` (`float32`) |
| Overflow | `3.0e38 + 3.0e38`, `max * 2` |

### 9.3 What cross-backend equality is achievable

The brief for this document warned against promising bitwise cross-backend
equality before establishing that it is achievable. It was therefore measured
rather than assumed.

**Integers.** Fixed-width two's-complement arithmetic is exact. Bitwise
equality across backends is required and trivially achievable.

**Floating point.** IEEE 754 requires addition, subtraction, multiplication
and division to be correctly rounded, so the result is uniquely determined.
Measuring NumPy on the CPU against CuPy on the GPU over ~394,000 random
`float32` operand pairs per operation:

| Restriction | add | sub | mul | div |
| --- | --- | --- | --- | --- |
| All finite operands | 198359/198486 | 198359/198486 | 189224/198486 | 189114/198486 |
| **Normal operands and normal result** | **393651/393651** | **393648/393648** | **326917/326917** | **328447/328447** |

Every single disagreement involved a **subnormal** — a subnormal operand
treated as zero on the device, or a subnormal result flushed to zero. With
subnormals excluded, CPU and GPU agreed **bit for bit on every pair**, for all
four operations.

The conformance requirement follows from the measurement rather than from
optimism:

- **Required: exact bitwise equality** across backends for all results that
  are normal, zero or infinite, for `+`, `-`, `*` and `/`, on both floating
  dtypes.
- **NaN**: required to be NaN, with payload and sign unspecified and not
  compared.
- **Subnormal results**: pending [O4](#o4-subnormal-handling-on-cuda). Until
  that is decided, a conformance test must either exclude subnormal results or
  record the CUDA divergence as a known, specified exception.

**No numerical tolerance is specified, because none is needed.** A tolerance
would be an admission that the result is not determined; for these four
operations it is. Tolerances belong to operations IEEE 754 does not require to
be correctly rounded, which this document does not cover.

### 9.4 Tests that will need revisiting

`tests/backend/_support.py` defines `NumPyParityTestCase`, whose
`assertOperationParity` evaluates an expression on the Python backend and
asserts the NumPy backend matches it. That is the "Python is the reference"
arrangement expressed as a test helper.

It remains a useful *cross-check* — two independent implementations agreeing
is evidence — but it must stop being the definition of correctness. Conformance
tests must assert against values written in the test, derived from this
document, not against whatever the Python backend produces.

---

## 10. Migration and compatibility

> Adopting this specification is a **deliberate semantic change**, not a
> bug fix. It will change observable results.

### 10.1 Known differences from current behaviour

Each row was verified against the installed package on all three backends.

| Area | Current behaviour | Specified behaviour |
| --- | --- | --- |
| Integer overflow | Raises `OverflowError: Python int too large to convert to C long` on every backend | Wraps: `int32` `2147483647 + 1` → `-2147483648` |
| Integer intermediates | NumPy computes in `object` arrays of Python integers to keep arbitrary precision | Native fixed-width arithmetic |
| CUDA integers | Rejected in `_operand`; every integer operation falls back to the Python kernel | Executes natively |
| `float32` working precision | Operands widened to `float64` unconditionally, then narrowed | Computed at declared precision |
| `float32` overflow | NumPy and CUDA decline, and the operation falls back to the Python kernel | Produces `inf` on the native kernel |
| Result storage | A `float32` result is stored as `float32` when accelerated | Unchanged |
| Infinity and NaN propagation | Correct and identical on all three backends | Unchanged |

The integer-overflow change is the largest. Today the error surfaces from the
Python `array` conversion, which is why it appears identically on all three
backends: every path ultimately stores through the same typed buffer.

### 10.2 What may depend on the old behaviour

- **Code relying on `OverflowError`** as a signal that an integer computation
  left its range will stop receiving it. Wraparound is silent by design.
- **Numerical results in `float32`** may change in the last ulp wherever an
  evaluation strategy previously carried a binary64 intermediate across more
  than one operation. Per single operation, results will not change
  ([section 5.5](#55-why-declared-precision-matters)).
- **Tests asserting current behaviour.** No test currently asserts
  `OverflowError`; that was checked. Tests that compare a backend against the
  Python backend rather than against specified values
  ([section 9.4](#94-tests-that-will-need-revisiting)) will need review.
- **Benchmark baselines.** Stored measurements predate the change. Numbers
  taken before and after are not comparable, because the fallbacks being
  removed are a large part of what the current numbers measure.

### 10.3 What has not happened

**No implementation change has been made.** At the time of writing, the
package behaves as the "Current behaviour" column describes. This document
records an approved target, and an implementation task will enforce it.

---

## 11. Open specification questions

These must be decided before the behaviour they govern is implemented or
changed. An implementation task must treat each as **do not touch**.

### O1. Mixed-dtype promotion

What is the result dtype of an operation on operands of different dtypes?
Sub-questions: `int32` with `int64`; an integer dtype with a floating dtype;
`float32` with `float64`; whether promotion depends on operand values; how
tensor–scalar arithmetic, where the scalar has no declared dtype, is treated;
and whether the existing `result_dtype` behaviour is retained, amended or
replaced. See [section 6.2](#62-undecided-mixed-dtypes).

### O2. Division by zero

Does floating division by zero continue to raise `ZeroDivisionError`, or adopt
the IEEE 754 result (signed infinity, and NaN for `0/0`)? If IEEE semantics
are adopted, are the exception flags observable, and what happens to the
existing guard in the operation layer? See
[section 7.2](#72-undecided-division-by-zero).

### O3. Integer division

What does `/` on two integer tensors mean, and what dtype does it produce?
Is a separate floor-division operation wanted? What happens on division by
zero under integer semantics, where there is no infinity to return? See
[section 7.3](#73-undecided-integer-division).

### O4. Subnormal handling on CUDA

Must the CUDA backend implement IEEE gradual underflow, or is flush-to-zero an
accepted, documented deviation? This is the only measured source of
cross-backend disagreement ([section 9.3](#93-what-cross-backend-equality-is-achievable)).
Deciding it requires knowing whether the device kernels can be built with
denormal support at acceptable cost, and whether the affected magnitudes
(below `1.1754944e-38` in `float32`) matter for the workloads this package
serves.

### O5. Dtypes outside this contract

`int16`, `int8` and `uint8` are defined in `tensors/dtype.py` but not covered
here. `uint8` in particular is **unsigned**, so the wraparound rule of
[section 4.2](#42-the-wraparound-rule) does not apply to it as written: the
unsigned form is \(r \bmod 2^{w}\).

### O6. Fused evaluation

The fusion path evaluates several operations as one kernel. Whether a fused
kernel must produce the same result as the equivalent sequence of individual
operations — which forbids retaining a wider intermediate, and forbids
contracting a multiply and an add into a fused multiply-add — is not decided
here. [Section 5.5(b)](#55-why-declared-precision-matters) shows that the
difference is observable in 25% of random `float32` triples, so this is a
question with consequences, not a formality.

### O7. Scope of "no fallback"

[Section 8.3](#83-legitimate-fallback) requires a fallback to be explicit and
observable, but does not define the mechanism. Whether that means a warning, a
counter, a structured diagnostic, or a benchmark-visible label is undecided.

---

## Related documents

- [Numerical backends](backends.md) — backend selection, native storage,
  kernel coverage, the workload policy and the current behaviour contract.
- [Tensor memory model](memory-model.md) — logical shape, strides and physical
  storage.
- [Package structure](package-structure.md) — where operations, dispatch and
  the backend kernels live.
