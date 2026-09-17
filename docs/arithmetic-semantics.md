# Arithmetic semantics

## 1. Status and scope

> **Status: implemented for `+`, `-`, `*` and `/`; specification elsewhere.**
>
> Addition, subtraction, multiplication and true division follow this document
> on the Python, NumPy and CUDA backends, in eager execution, graph replay,
> differentiation and fusion. Passages marked *Current behaviour* describe what
> the implementation did **before** that work and are kept because
> [section 10](#10-migration-and-compatibility) explains the migration away
> from it.
>
> Every other operation — `**`, comparisons, `where`, `clip`, `concat`,
> reductions, matmul, the elementwise maths functions — still uses the older
> promotion in `tensors/dtype.py` `result_dtype` and is **not** governed by
> this document yet. Extending it to them is separate work.
>
> Two gaps within the implemented scope are recorded rather than closed:
> the execution-location observability API of
> [backends.md](backends.md#observability) does not exist, and the VJP
> execution-location requirement of [autodiff.md](autodiff.md) is unmet for
> the kernels named in [section 10.3](#103-implementation-paths-requiring-review).

This document is the **authoritative numerical specification**. Backend
selection and execution requirements are in [Numerical backends](backends.md);
graph execution and optimisation requirements are in
[Automatic differentiation](autodiff.md). Those documents reference this one
rather than restating it.

### In scope

Elementwise **addition**, **subtraction**, **multiplication** and **division**
over the seven public scalar dtypes, for tensor operands and for Python scalar
operands, including mixed-dtype combinations and the promotion rules that
govern them.

| dtype | Kind | Typecode | Width |
| --- | --- | --- | --- |
| `uint8` | unsigned integer | `B` | 1 byte |
| `int8` | signed integer | `b` | 1 byte |
| `int16` | signed integer | `h` | 2 bytes |
| `int32` | signed integer | `i` | 4 bytes |
| `int64` | signed integer | `q` | 8 bytes |
| `float32` | IEEE 754 binary32 | `f` | 4 bytes |
| `float64` | IEEE 754 binary64 | `d` | 8 bytes |

These are exactly the `DataType` values exported from `tensors.dtype` and
re-exported in `tensors.__all__`; the list was audited against the package
rather than assumed.

### Out of scope

- **Floor division** and any integer-valued division operator
  ([section 7.4](#74-floor-division-is-a-separate-operation)).
- Comparison, reduction, linear-algebra and transcendental operations.
- Tensor construction and explicit casting, except to distinguish them from
  arithmetic ([section 4.5](#45-arithmetic-is-not-construction-or-casting)).
- Broadcasting, which decides *which* element pairs combine, not what a pair
  produces.
- Automatic differentiation, beyond the requirement that graph optimisation
  preserve these semantics ([section 8.5](#85-graph-optimisation)).

---

## 2. Architectural principles

These six principles govern this document, [backends.md](backends.md) and
[autodiff.md](autodiff.md) alike.

**P1. The declared dtype and this specification determine semantics.** Not the
values stored, not the backend, not the tensor's size, not the execution path
chosen for it.

**P2. Python, NumPy and CUDA are implementations. None is the semantic
authority.** Where a backend and this document disagree, the backend is wrong
— including the Python backend.

**P3. A supported operation behaves identically regardless of backend, tensor
size or execution path.** An operation must not change its results because a
tensor was small enough to take one path or large enough to take another.

**P4. Selecting a backend explicitly is an execution requirement, not a
preference.** `ts.set_backend("cuda")` means the operation runs on CUDA or
reports that it cannot. See [backends.md](backends.md).

**P5. Graph optimisation preserves the numerical semantics of the function it
optimises.** Fusion may remove allocations and memory traffic. It may not
change results. See [section 8.5](#85-graph-optimisation) and
[autodiff.md](autodiff.md).

**P6. Legacy behaviour is not preserved merely for compatibility when it
contradicts this architecture.** Adopting this specification introduces
deliberate breaking changes, listed in
[section 10](#10-migration-and-compatibility).

### 2.1 What this replaces

The package was built with the Python backend's behaviour as the definition.
The NumPy and CUDA kernels reproduce it, and decline when they cannot. The
cost is visible in the implementation:

- `tensors/backend/numpy/conversion.py` widens every floating operand to
  `numpy.float64` regardless of declared dtype, and computes integer
  arithmetic in `object` arrays of Python integers.
- `tensors/backend/cuda/conversion.py` refuses integer dtypes outright, so
  every integer operation falls back to the Python kernel.
- Both decline a `float32` result that would round to infinity, because the
  Python reference does not produce that overflow.
- Both force a host synchronisation on every division to test for a zero
  denominator, and on every narrowing float result to test for overflow.

Under this specification that machinery is unnecessary, and the direction of
authority reverses.

### 2.2 Independence from Python and NumPy

The specification does not inherit from Python's built-in numeric types.
Python's `int` is arbitrary-precision and its `float` is always binary64;
neither models a typed tensor library, where the declared width *is* the
contract. **No requirement in this document exists to reproduce the behaviour
of Python's built-in `float` or `int`.**

Where this document and NumPy agree, it is because the same standard was
applied, not because NumPy was consulted.

---

## 3. Dtypes and representable values

### 3.1 Integer dtypes

| dtype | Signed | Width \(w\) | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| `uint8` | no | 8 | `0` | `255` |
| `int8` | yes | 8 | `-128` | `127` |
| `int16` | yes | 16 | `-32768` | `32767` |
| `int32` | yes | 32 | `-2147483648` | `2147483647` |
| `int64` | yes | 64 | `-9223372036854775808` | `9223372036854775807` |

Signed types are two's complement. These match `_INTEGER_LIMITS` in
`tensors/dtype.py`.

### 3.2 Floating dtypes

| dtype | Format | Precision \(p\) | Max finite | Min normal | Min subnormal | Epsilon |
| --- | --- | --- | --- | --- | --- | --- |
| `float32` | binary32 | 24 bits | `3.4028235e+38` | `1.1754944e-38` | `1e-45` | `1.1920929e-07` |
| `float64` | binary64 | 53 bits | `1.7976931348623157e+308` | `2.2250738585072014e-308` | `5e-324` | `2.220446049250313e-16` |

Each constant above is the shortest decimal that round-trips exactly to the
intended value in its own format; this was verified.

Each format also represents signed zeros, signed infinities and NaNs.

### 3.3 Exact integer representation in floating formats

A binary format with precision \(p\) represents every integer in
\([-2^{p},\, 2^{p}]\) exactly, and \(2^{p}+1\) is the first positive integer it
cannot. This bound decides integer–floating promotion
([section 6.3](#63-integer-with-floating)), because promotion must accommodate
**every** value of the operand dtype.

The bound is sufficient, not necessary, for a *particular* integer. Beyond
\(2^{p}\) an integer is still exact when it has enough trailing zero bits:
\(2^{25}\) is exact in `float32` and \(2^{54}\) is exact in `float64`,
while \(2^{24}+1\) and \(2^{53}+1\) are not. That distinction matters for a
single scalar literal, where [rule S4](#65-python-scalars) applies the exact
test rather than the interval.

| Integer dtype | Exact in `float32` (\(2^{24}\)) | Exact in `float64` (\(2^{53}\)) |
| --- | --- | --- |
| `uint8`, `int8`, `int16` | yes | yes |
| `int32` | **no** (\(2^{31}-1 > 2^{24}\)) | yes |
| `int64` | **no** | **no** (\(2^{63}-1 > 2^{53}\)) |

Verified: `float32(2147483647)` is `2147483648`; `float64(9223372036854775807)`
is `9223372036854775808`.

### 3.4 Storage residency is not semantics

A `float32` tensor may live in a Python `array('f')`, a `numpy.ndarray`, or a
device-resident `cupy.ndarray`. All three must produce the same values. A
backend may use any internal representation that yields the specified result,
but **a result of declared dtype `D` is stored as `D`** — a `float32` result is
a `float32` tensor, not a `float64` buffer carrying a `float32` label.

Where results physically reside, and which transfers are permitted, is
specified in [backends.md](backends.md).

---

## 4. Integer arithmetic

### 4.1 Rules

For addition, subtraction and multiplication where **both operands have the
same integer dtype**:

1. **The result has that dtype.**
2. **Arithmetic is fixed-width with wraparound.**
3. **Overflow does not raise**, and does not promote to a wider dtype.
4. **Results are bit-identical across every backend.**

### 4.2 The wraparound rule

Let \(r\) be the **mathematical** result of the operation: the exact value over
the integers, computed with no width limit, before any wrapping.

For a **signed** dtype of width \(w\):

$$
\operatorname{wrap}_w(r) \;=\; \bigl((r + 2^{\,w-1}) \bmod 2^{\,w}\bigr) - 2^{\,w-1}
$$

For an **unsigned** dtype of width \(w\):

$$
\operatorname{wrap}^{\mathrm{unsigned}}_w(r) \;=\; r \bmod 2^{\,w}
$$

In both, \(\bmod\) is the non-negative remainder, so the result is defined for
negative \(r\). Equivalently, each returns the unique value in the dtype's
range congruent to \(r\) modulo \(2^w\). When \(r\) is already representable,
the rule returns \(r\), so it describes ordinary arithmetic as well as
overflow.

`uint8` is the only unsigned public dtype, so the unsigned form applies to it
alone.

> \(r\) is definitional, not an implementation requirement. It says *which*
> value must be stored, not how to compute it — see
> [section 4.4](#44-implementation-freedom).

### 4.3 Worked examples

Every row was checked against the formula **and** against native fixed-width
arithmetic; the two agree in all cases.

| dtype | Expression | Mathematical \(r\) | Stored result |
| --- | --- | --- | --- |
| `uint8` | `255 + 1` | `256` | `0` |
| `uint8` | `0 - 1` | `-1` | `255` |
| `uint8` | `16 * 16` | `256` | `0` |
| `int8` | `127 + 1` | `128` | `-128` |
| `int8` | `-128 - 1` | `-129` | `127` |
| `int8` | `-128 * -1` | `128` | `-128` |
| `int16` | `32767 + 1` | `32768` | `-32768` |
| `int16` | `-32768 - 1` | `-32769` | `32767` |
| `int16` | `256 * 256` | `65536` | `0` |
| `int32` | `2147483647 + 1` | `2147483648` | `-2147483648` |
| `int32` | `-2147483648 - 1` | `-2147483649` | `2147483647` |
| `int32` | `1000000 * 1000` | `1000000000` | `1000000000` |
| `int32` | `65536 * 65536` | `4294967296` | `0` |
| `int32` | `2147483647 * 2147483647` | `4611686014132420609` | `1` |
| `int32` | `-2147483648 * -1` | `2147483648` | `-2147483648` |
| `int64` | `9223372036854775807 + 1` | `9223372036854775808` | `-9223372036854775808` |
| `int64` | `-9223372036854775808 - 1` | `-9223372036854775809` | `9223372036854775807` |
| `int64` | `4294967296 * 4294967296` | `18446744073709551616` | `0` |

Reading the first `int32` row through the formula, with \(w = 32\):

$$
\operatorname{wrap}_{32}(2147483648)
= \bigl((2147483648 + 2^{31}) \bmod 2^{32}\bigr) - 2^{31}
= 0 - 2147483648
= -2147483648
$$

Note `-128 * -1` in `int8` and `-2147483648 * -1` in `int32`: the negation of
the minimum is not representable, so the rule returns the minimum again. That
is a consequence of the rule, not an exception to it.

### 4.4 Implementation freedom

An implementation may produce the specified value by any means: native
fixed-width machine arithmetic that wraps in hardware, computation in a wider
type followed by reduction, or masking the low \(w\) bits.

**Arbitrary-precision arithmetic is not required.** The current NumPy path
computes in `object` arrays of Python integers to reproduce Python's
arbitrary-precision intermediates. Under this specification that is
unnecessary: native `uint8`, `int8`, `int16`, `int32` and `int64` arithmetic
already wraps exactly as the rule requires, which is also the cheapest correct
implementation on every backend.

### 4.5 Arithmetic is not construction or casting

Wraparound is a rule about **arithmetic**. It does not apply to:

- **Tensor construction.** `ts.Tensor([256], dtype=ts.uint8)` supplies a value
  the dtype cannot represent. That is a caller error, and it raises.
- **Explicit casting.** `tensor.astype(ts.uint8)` on a value outside `uint8`
  raises for integer targets.

*Current behaviour (verified):* both already raise `OverflowError`, with
messages from the underlying typed buffer. Float→float casting that exceeds
the target range yields `inf`, and float→integer casting truncates toward
zero. **This specification changes none of it.** Only arithmetic wraps.

The distinction is deliberate. An out-of-range *literal* is almost always a
mistake; an out-of-range *arithmetic result* is a defined consequence of
finite width.

---

## 5. Floating-point arithmetic

### 5.1 Rules

1. **Same-dtype arithmetic preserves the dtype** for `+`, `-`, `*`, `/`.
2. **Operations are performed in the declared precision.** `float32` operands
   are not unconditionally widened to `float64`.
3. **Overflow produces a signed infinity.** It does not raise, and does not
   cause a backend to decline and defer to another.
4. **Infinities and NaNs propagate** as IEEE 754 specifies.
5. **Signed zero is preserved.**
6. **Results are stored in their declared dtype.**

The foundation is **IEEE 754**: binary32 for `float32`, binary64 for `float64`.

### 5.2 Rounding

Addition, subtraction, multiplication and division are **correctly rounded**:
the delivered result is the representable value nearest the exact mathematical
result, as if computed with unbounded range and precision and rounded once.

The rounding mode is **round-to-nearest, ties-to-even**. No other mode is
specified and an implementation must not select one.

Correct rounding is what makes cross-backend agreement attainable: for these
four operations the result is *uniquely determined* by the operand values, the
dtype and the rounding mode. There is no latitude for a conforming
implementation to differ. This is not true of transcendental functions, which
IEEE 754 does not require to be correctly rounded and which this document does
not cover.

| dtype | Expression | Result | Why |
| --- | --- | --- | --- |
| `float32` | `1.0 + 2**-24` | `1.0` | exactly half an ulp; ties-to-even picks the even significand |
| `float32` | `1.0 + 2**-23` | `1.0000001` | one ulp above `1.0`, exactly representable |

### 5.3 Infinities, NaNs and signed zero

| Case | Result |
| --- | --- |
| `3.0e38 + 3.0e38` (`float32`) | `inf` |
| `max * 2` | `inf` |
| `inf + 1.0` | `inf` |
| `inf - inf` | `nan` |
| `inf / inf` | `nan` |
| `0.0 * inf` | `nan` |
| `nan + 1.0` | `nan` |
| `nan == nan` | `False` |
| `0.0 + -0.0` | `+0.0` |
| `-0.0 + -0.0` | `-0.0` |
| `-0.0 == 0.0` | `True` |

- **Infinities** are produced by overflow and propagate. They are values, not
  errors.
- **NaN** propagates through every arithmetic operation. **NaN payloads and
  the NaN sign bit are not specified**; conformance tests must treat all NaNs
  as equivalent and must not compare NaN bit patterns.
- **Signed zero** follows IEEE 754 and is observable through division
  (`1.0 / -0.0` is `-inf`). It must not be discarded.

*Current behaviour (verified):* infinities and NaNs already propagate
correctly and identically on all three backends for addition and subtraction.
What changes is overflow, which must stop triggering a fallback.

### 5.4 Subnormals: gradual underflow is required

**IEEE 754 gradual underflow is required**, for both subnormal **operands** and
subnormal **results**. Flush-to-zero is not an accepted deviation.

| Expression (`float32`) | Required result |
| --- | --- |
| `min_normal / 2` | `5.877472e-39` (subnormal) |
| `min_subnormal / 2` | `0.0` (genuine underflow to zero) |
| `1e-40 * 1.0` | `1e-40` (subnormal operand preserved) |

*Current behaviour and capability (verified on CuPy 14.2, CUDA 13.0, compute
capability 8.6):*

| Path | Subnormal preserved? |
| --- | --- |
| `float64`, every CUDA path tested | **yes** |
| Host→device transfer and a pure copy kernel, `float32` | **yes** |
| `cupy.multiply`, `cupy.divide` ufunc, `float32` | no — flushed |
| `cupy.ElementwiseKernel` with `z = x * y`, `float32` | no — flushed |
| `cupy.RawModule`, `float32`, with and without `--ftz=false` | no — flushed |
| **Raw kernel or `ElementwiseKernel` with inline PTX `mul.rn.f32`, `add.rn.f32`, `sub.rn.f32`, `div.rn.f32`** | **yes — bit-identical to the CPU** |

This is a **CuPy code-generation default, not a hardware limitation.** The
device supports binary32 gradual underflow: inline PTX `mul.rn.f32` and
`add.rn.f32` reproduce the CPU result exactly, including subnormal operands and
subnormal results. NVRTC emits the IEEE instruction `mul.f32` by default
(`--ftz=true` emits `mul.ftz.f32`), so the flush is introduced below PTX, in
the path CuPy uses to reach a loadable module.

**A conforming CUDA implementation is therefore achievable**, by generating
the `.rn` PTX forms for `float32` elementwise arithmetic instead of relying on
CuPy's default code generation. Until such an implementation exists,
[section 8.4](#84-when-a-backend-cannot-conform) governs what strict CUDA
selection must do.

### 5.5 Why declared precision matters

The naive argument for this requirement is wrong, and an implementer who
believes the wrong version will fix the wrong thing.

#### 5.5.1 A single widened operation is not observably different

Let \(\circ\) be one of \(+, -, \times, \div\) on two binary32 operands.
Compute \(\circ\) exactly, round to binary64 (round-to-nearest-even), then
round that to binary32 (round-to-nearest-even). This **double rounding is
innocuous** — it yields the correctly-rounded binary32 result — under the
classical condition on the two precisions:

$$
p_{\text{wide}} \;\ge\; 2\,p_{\text{narrow}} + 2
$$

For binary32 inside binary64: \(p_{\text{narrow}} = 24\),
\(p_{\text{wide}} = 53\), and \(53 \ge 2 \times 24 + 2 = 50\). The condition
holds, so a **single** widened binary32 operation is bit-identical to native
binary32.

Conditions and limits that must accompany that statement:

- **The condition is on precision, and it is not general.** For binary64
  inside x87 80-bit extended, \(p_{\text{narrow}} = 53\) and
  \(p_{\text{wide}} = 64\), while the condition demands 108. Double rounding
  through 80-bit extended is **not** innocuous for `float64`. This reasoning
  must not be transferred to `float64`.
- **Overflow is safe for a single operation.** A binary64 intermediate beyond
  binary32's finite range rounds to \(\pm\infty\) on the second rounding,
  matching native binary32.
- **Underflow is safe for a single operation**, because binary64's exponent
  range extends far below binary32's, so the intermediate cannot itself
  underflow; the binary32 subnormal result is reached by one rounding of an
  exact-enough intermediate.
- **It applies to exactly one operation.** The condition says nothing about a
  sequence.

Targeted sweeps over random and subnormal-range operands found no
disagreement, which is consistent with the theorem. **That agreement is not
the justification** — the precision condition is. Sampling cannot establish a
universal claim.

#### 5.5.2 Why widening is nevertheless wrong

**(a) The intermediate range differs.** binary64's exponent range is far
wider, so an intermediate that must overflow in binary32 survives:

```text
a = 1e30, b = 1e20, c = 1e25   (all float32)

per-operation float32 : (a * b) / c  ->  inf
float64 throughout    : (a * b) / c  ->  1.0000001e+25
```

A `float32` computation that overflows must overflow. Carrying the
intermediate in binary64 silently rescues it — a different function, not a
more accurate one.

**(b) Precision is retained across operations.** Section 5.5.1 covers *one*
operation. Feed a widened intermediate to the next operation without rounding
it back, and results diverge. Across 200,000 random `float32` triples,
`(a * b) + c` computed per-operation in binary32 differed from the same
expression evaluated entirely in binary64 in **25%** of cases:

```text
a = 38.24823, b = 280.5834, c = -0.4536956

per-operation float32 : 10731.364   (bits 0x4627ad75)
float64 throughout    : 10731.365   (bits 0x4627ad76)
```

This is why "we round the final result to float32" is not a sufficient
defence, and why [section 8.5](#85-graph-optimisation) constrains fusion.

**(c) It costs what it was meant to save.** Widening every operand allocates
and converts a buffer of twice the width and converts back. On the accelerated
backends that is the dominant cost of a small operation, and per operation it
buys nothing.

The requirement is therefore stated in terms of what is observable: a
`float32` operation must produce the correctly-rounded binary32 result, must
overflow when binary32 overflows, must underflow gradually as binary32 does,
and must produce a `float32` result stored as `float32`. An implementation
that achieves this by widening a single operation is conforming but wasteful;
one that widens across an expression is not conforming.

---

## 6. Result dtypes and promotion

Promotion depends on **operand dtypes only**. It never depends on the values
stored in individual tensor elements. (Python scalars, which have no dtype, are
specified separately in [section 6.5](#65-python-scalars).)

### 6.1 Principles

- **P-a.** Same-dtype operands preserve their dtype, except where an operation
  defines a different result type (division does; see
  [section 7](#7-division)).
- **P-b.** Mixed integer operands promote to the **smallest public integer
  dtype whose range contains both operand ranges**.
- **P-c.** Mixed floating operands promote to the **wider** floating dtype.
- **P-d.** Integer with floating promotes to the **narrowest floating dtype
  that represents every value of the integer dtype exactly**
  ([section 3.3](#33-exact-integer-representation-in-floating-formats)), and
  never narrower than the floating operand.
- **P-e.** If no public dtype satisfies the rule, the operation **raises and
  requires an explicit cast**. It must not silently choose a lossy result.

### 6.2 The promotion table

Rows and columns are operand dtypes; cells are the result dtype for `+`, `-`
and `*`. **`cast`** means the operation raises and the caller must cast
explicitly.

|  | `uint8` | `int8` | `int16` | `int32` | `int64` | `float32` | `float64` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **`uint8`** | `uint8` | `int16` | `int16` | `int32` | `int64` | `float32` | `float64` |
| **`int8`** | `int16` | `int8` | `int16` | `int32` | `int64` | `float32` | `float64` |
| **`int16`** | `int16` | `int16` | `int16` | `int32` | `int64` | `float32` | `float64` |
| **`int32`** | `int32` | `int32` | `int32` | `int32` | `int64` | **`float64`** | `float64` |
| **`int64`** | `int64` | `int64` | `int64` | `int64` | `int64` | **`cast`** | **`cast`** |
| **`float32`** | `float32` | `float32` | `float32` | **`float64`** | **`cast`** | `float32` | `float64` |
| **`float64`** | `float64` | `float64` | `float64` | `float64` | **`cast`** | `float64` | `float64` |

The table is complete (49 cells) and **symmetric**, which was checked
mechanically: `promote(a, b) == promote(b, a)` for every pair. Symmetry is
required because `+` and `*` are commutative, and it would be incoherent for
`-` and `/` to promote differently from `+`.

Worked justifications for the non-obvious cells:

- **`uint8` with `int8` → `int16`.** The union of \([0, 255]\) and
  \([-128, 127]\) is \([-128, 255]\). Neither `uint8` nor `int8` contains it;
  `int16` is the smallest public dtype that does (P-b).
- **`int32` with `float32` → `float64`.** `float32` represents integers
  exactly only to \(2^{24}\), and `int32` reaches \(2^{31}-1\). `float64`
  reaches \(2^{53}\) and contains `int32` exactly (P-d).
- **`int64` with either floating dtype → `cast`.** \(2^{63}-1\) exceeds
  \(2^{53}\), so neither public floating dtype represents every `int64` value.
  No public dtype satisfies P-d, so P-e applies.

### 6.3 Integer with floating

P-d exists to prevent silent, avoidable integer precision loss. `int32 +
float32` promoting to `float32` would silently round operands above
\(2^{24}\); promoting to `float64` does not.

The `int64` cells are the deliberate consequence. There is no public dtype
that holds every `int64` value and every floating value, so the specification
refuses rather than choosing a lossy answer. The caller writes what they meant:

```python
big.astype(ts.float64) + scale        # accept the rounding, explicitly
```

`Tensor.astype` already exists and is the required mechanism; no new API is
needed for this.

### 6.4 What is not promoted

Promotion never inspects element values. `int64` tensors holding only small
values still promote as `int64`. This keeps the result dtype a static property
of the expression, which graph compilation and replay require.

### 6.5 Python scalars

A Python scalar has no declared dtype, so it cannot participate in the table
above. It is **weakly typed**: it is converted to the tensor's dtype, and the
result has the tensor's dtype. What "converted" permits depends on the kind of
scalar and the kind of target, so the rule is stated once per combination
rather than as a single phrase.

Throughout, "converts" means the operation proceeds and the result has the
tensor's dtype; "raises" means the operation raises `TypeError` and the caller
must supply a typed operand or cast the tensor.

> **S1 — Python `int` with an integer tensor.** Converts when the value lies
> within the tensor dtype's representable range
> ([section 3.1](#31-integer-dtypes)). The conversion is exact; no rounding is
> performed and no wraparound is applied. A value outside the range raises.
>
> **S2 — Python `float` with an integer tensor.** Converts only when the value
> is *integral* — that is, it has no fractional part — **and** lies within the
> tensor dtype's range. **A non-integral Python float is never implicitly
> converted to an integer dtype**; it raises.
>
> **S3 — Python `float` with a floating tensor.** Converts by **rounding** to
> the target format under round-to-nearest, ties-to-even
> ([section 5.2](#52-rounding)). Permitted when the rounded value is finite,
> and also when the literal is itself an infinity or a NaN. A finite literal
> whose magnitude exceeds the target's range — one that would round to
> infinity — raises. Rounding to a subnormal, or underflowing to zero, is
> ordinary rounding and is permitted.
>
> **S4 — Python `int` with a floating tensor.** Converts **if and only if
> that particular integer is exactly representable** in the target format. An
> integer that is not raises, rather than being silently rounded.
>
> Writing \(|n| = m \cdot 2^{k}\) with \(m\) odd, a non-zero integer \(n\)
> is exactly representable when \(m\) needs at most \(p\) bits — \(p = 24\)
> for `float32`, \(p = 53\) for `float64` — and \(|n|\) does not exceed the
> format's largest finite value. Equivalently: converting \(n\) to the target
> format and back yields \(n\).
>
> **The interval \([-2^{p},\, 2^{p}]\) is sufficient but not necessary.**
> Every integer inside it is representable, which is why
> [section 3.3](#33-exact-integer-representation-in-floating-formats) uses it
> to decide *dtype* promotion, where every value of the operand dtype must be
> accommodated. A *single* integer outside it may still be representable if it
> has enough trailing zero bits: \(2^{25}\) is exactly representable in
> `float32` because its odd part is \(1\), while \(2^{24}+1\) is not because
> its odd part needs 25 bits.

Only S3 rounds. S1, S2 and S4 require the value to be representable exactly,
because in those three cases rounding would discard information the caller
wrote down literally: an integer target cannot hold a fraction, and an integer
a floating target cannot represent is precisely the precision loss
[section 6.3](#63-integer-with-floating) exists to prevent.

Note the asymmetry between S4 and the promotion table. S4 asks whether *this
integer* is representable; promotion asks whether *every value of a dtype* is.
A `float32` tensor therefore accepts the scalar \(2^{25}\), while `int32`
with `float32` still promotes to `float64` — because some `int32` values are
not representable in `float32`, even though that one is.

The result dtype is the tensor's dtype in every case. Using the scalar's
*value* here is not a violation of [section 6.4](#64-what-is-not-promoted):
that rule constrains **tensor** operands, whose elements vary. A Python scalar
is a single literal, and its value is the only information it carries.

| Expression | Rule | Result |
| --- | --- | --- |
| `float32_tensor + 2.0` | S3 — rounds to `2.0`, finite | `float32` |
| `float32_tensor + 0.1` | S3 — rounds to the nearest `float32` | `float32` |
| `float32_tensor + 1e-60` | S3 — underflows to `0.0`, ordinary rounding | `float32` |
| `float32_tensor + float("inf")` | S3 — an infinity is representable | `float32` |
| `float32_tensor + 1e300` | S3 — finite, but beyond `float32` range | raises `TypeError` |
| `float32_tensor + 2` | S4 — exactly representable | `float32` |
| `float32_tensor + 2**24` | S4 — the largest integer with every predecessor representable | `float32` |
| `float32_tensor + (2**24 + 1)` | S4 — odd, needs 25 bits | raises `TypeError` |
| `float32_tensor + 2**25` | S4 — odd part is \(1\); outside the interval, still exact | `float32` |
| `float32_tensor + (2**25 + 1)` | S4 — odd, needs 26 bits | raises `TypeError` |
| `float64_tensor + 2` | S4 — exactly representable | `float64` |
| `float64_tensor + (2**53 + 1)` | S4 — odd, needs 54 bits | raises `TypeError` |
| `float64_tensor + 2**54` | S4 — odd part is \(1\); outside the interval, still exact | `float64` |
| `int32_tensor + 2` | S1 — within `int32` range | `int32` |
| `int64_tensor + 1` | S1 — within `int64` range | `int64` |
| `uint8_tensor + 255` | S1 — within `uint8` range | `uint8` |
| `uint8_tensor + 256` | S1 — outside `uint8` range | raises `TypeError` |
| `uint8_tensor + (-1)` | S1 — outside `uint8` range | raises `TypeError` |
| `int32_tensor + 3.0` | S2 — integral and within range | `int32` |
| `int32_tensor + 3.5` | S2 — not integral | raises `TypeError` |
| `int32_tensor + True` | rejected — see below | raises `TypeError` |

Two consequences worth stating plainly:

- **`int32_tensor + 3.5` raises** (S2). Under the previous architecture it
  promoted to `float64`. Refusing is the consistent reading of P-e: the caller
  asked to combine an integer tensor with a non-integral value, and which dtype
  they wanted is genuinely ambiguous. `t.astype(ts.float64) + 3.5` says it.
- **A scalar never widens the result.** No scalar causes promotion to a wider
  dtype; it either converts to the tensor's dtype or raises. This is what keeps
  the result dtype of an expression a static property, as
  [section 6.4](#64-what-is-not-promoted) requires.

Python `bool` is **not** accepted as a numeric scalar, even though `bool` is a
subclass of `int` in Python, and there is no public boolean dtype. Passing one
raises.

### 6.6 Arithmetic after promotion

Promotion selects the result dtype. **The arithmetic is then performed in that
dtype**, under the rules of [section 4](#4-integer-arithmetic) or
[section 5](#5-floating-point-arithmetic). So `int8 + int8` wraps at 8 bits,
while `int8 + int16` promotes to `int16` first and then wraps at 16 bits.

---

## 7. Division

Division is specified separately because its result dtype does not follow the
rule for `+`, `-` and `*`.

### 7.1 Floating division

For same-dtype floating operands:

| Left | Right | Result dtype |
| --- | --- | --- |
| `float32` | `float32` | `float32` |
| `float64` | `float64` | `float64` |

Division is correctly rounded under round-to-nearest ties-to-even, exactly as
[section 5.2](#52-rounding) specifies. Mixed floating operands follow
[section 6.2](#62-the-promotion-table).

### 7.2 Division by zero

**Floating-point division adopts IEEE 754 results. It does not raise.**

| Expression | Result |
| --- | --- |
| `1.0 / +0.0` | `+inf` |
| `1.0 / -0.0` | `-inf` |
| `-1.0 / +0.0` | `-inf` |
| `-1.0 / -0.0` | `+inf` |
| `0.0 / 0.0` | `nan` |
| `inf / inf` | `nan` |
| `0.0 / inf` | `+0.0` |
| `inf / 0.0` | `+inf` |

**`ZeroDivisionError` must not be raised for floating-point division.**

**A backend must not introduce a host synchronisation to detect zero
denominators.** The IEEE result is produced by the hardware; there is nothing
to check.

*Current behaviour (verified):* floating division by zero raises
`ZeroDivisionError`. The guard sits in the operation layer
(`tensors/operations/arithmetic/divide.py`) and is mirrored in all three
backend kernels. On CUDA it is implemented as
`bool(cupy.any(right_array == 0))`, which forces a host synchronisation on
**every** division. Removing it is both a semantic change and a performance
change.

**Integer division by zero raises `ZeroDivisionError`.** There is no integer
infinity and no integer NaN, so there is no value to deliver. This is the one
case where division raises.

#### IEEE exception flags are not public API

IEEE 754 defines sticky status flags — *divide-by-zero*, *invalid*,
*overflow*, *underflow*, *inexact* — raised alongside these results.

**This refactor does not make those flags part of the public API.** No
accessor, no trapping mode, no per-operation status. An implementation may
leave them in whatever state the underlying hardware or library produces; no
behaviour depends on reading them. Exposing them would be a separate, additive
API decision.

The distinction matters: "division by zero raises the *divide-by-zero flag*"
is a statement about IEEE 754, and "division by zero raises a *Python
exception*" is a statement about this package. Under this specification the
first happens invisibly and the second does not happen at all for floating
division.

### 7.3 Integer true division

`/` is **true division**: the result is a floating-point value, never an
integer.

> **Rule D.** Integer true division is supported when the operand dtypes
> promote to a floating dtype under P-d
> ([section 6.2](#62-the-promotion-table)) — that is, when a public floating
> dtype represents every value of both integer operand dtypes exactly.
> Otherwise the operation raises and requires an explicit cast.

Applying the table to every public integer dtype:

| Left | Right | `/` result dtype | Why |
| --- | --- | --- | --- |
| `uint8` | `uint8` | `float32` | `uint8` exact in `float32` |
| `int8` | `int8` | `float32` | exact in `float32` |
| `int16` | `int16` | `float32` | exact in `float32` |
| `uint8` | `int8` | `float32` | promotes to `int16`, exact in `float32` |
| `uint8` | `int16` | `float32` | promotes to `int16` |
| `int8` | `int16` | `float32` | promotes to `int16` |
| `int32` | `int32` | `float64` | `int32` needs \(2^{53}\) |
| `int16` | `int32` | `float64` | promotes to `int32` |
| `uint8` | `int32` | `float64` | promotes to `int32` |
| `int64` | anything integer | **`cast`** | `int64` exact in no public floating dtype |
| any integer | `int64` | **`cast`** | as above |

So integer true division is supported for every public integer dtype except
`int64`, where it requires `x.astype(ts.float64) / y.astype(ts.float64)` and
the caller thereby accepts the rounding above \(2^{53}\).

*Current behaviour (verified):* `int32 / int32` and `int64 / int64` both
produce `float64`. Under this specification the first becomes `float64` (same
result, now by rule rather than by accident) and the second raises.

The rule means integer true division never silently converts `int64` to
`float64`, which is precisely the conversion that loses integer precision.

### 7.4 Floor division is a separate operation

Floor division, and any other integer-valued division, is a **separate
operation with its own future specification**. Nothing here decides its
semantics, its result dtype, or its behaviour on zero denominators or on the
`min // -1` boundary.

This task does not change any floor-division implementation and does not
introduce a new public operator. See [O1](#o1-floor-division).

---

## 8. Backend conformance

Backend *selection* and *execution* requirements live in
[backends.md](backends.md). This section states only what conformance means
numerically.

### 8.1 Requirements

1. **Python, NumPy and CUDA implement the same contract.** All are
   implementations; none is the definition.
2. **The Python backend is not the reference.** It is the simplest and most
   readable implementation, which makes it a useful cross-check, not an
   authority. Where it and this document disagree, it is wrong.
3. **A backend must not silently change dtype or arithmetic semantics**, and
   must not store a result in a representation wider than its declared dtype.
4. **A backend must not fall back to another merely because native arithmetic
   differs from Python's built-in semantics.** Native fixed-width wraparound
   and native binary32 arithmetic are *correct* here. They are the reason the
   current fallbacks exist, and they are not grounds for one.
5. **Native kernels should be used wherever they satisfy the contract.**

### 8.2 Conformance is not coverage

Requiring identical results **does not** assert that every operation is
natively implemented on every backend. It is not true today: integer
arithmetic on CUDA is rejected outright, and small workloads are routed to the
Python kernel by the policy in `tensors/backend/policy.py`.

### 8.3 Legitimate fallback

Under **automatic** selection, an implementation may execute an operation on a
backend other than the nominally selected one when it is genuinely unsupported
there. Such a fallback must **preserve the contract exactly** and must be
**observable**. See [backends.md](backends.md) for the selection modes and the
observability requirement.

A fallback is **not** legitimate when it exists to reproduce Python's
arithmetic semantics. That is the situation this specification removes.

### 8.4 When a backend cannot conform

If a backend cannot satisfy this specification for a supported operation, the
implementation must, under **explicit** backend selection, report that
operation as **unsupported** rather than silently producing a
non-conforming result or silently executing elsewhere.

This applies now to exactly one case, with the evidence in
[section 5.4](#54-subnormals-gradual-underflow-is-required): CUDA `float32`
elementwise arithmetic flushes subnormals under CuPy's default code
generation. Because a conforming implementation **is** achievable through
inline PTX, this is a temporary implementation gap rather than a permanent
capability limit, and closing it is part of the work
([section 10.4](#104-implementation-sequence), stage 3).

### 8.5 Graph optimisation

> **An optimisation must preserve the numerical result of the original
> sequence of typed operations.**

For an expression evaluated in dtype \(d\):

$$
t = \operatorname{round}_{d}(a \times b), \qquad
r = \operatorname{round}_{d}(t + c)
$$

A fused implementation must preserve **both** rounding boundaries. It must not
replace the pair with a fused multiply-add that rounds only once, because that
changes the result.

*Verified:* compiling `out = a * b + c` for CUDA with NVRTC's default
`--fmad=true` contracts the pair into an FMA and yields `1.0731365e+04`, while
`--fmad=false` yields `1.0731364e+04` — the per-operation result, bit-identical
to the CPU. The current fusion kernels are built with
`cupy.RawKernel(source, name)` and **no options**
(`tensors/backend/cuda/kernels/fusion/fused_elementwise.py`), so the default
applies and contraction is possible today.

Fusion may eliminate intermediate allocations and memory traffic. It may not
alter observable numerical semantics. The consequences for replay and backend
switching are in [autodiff.md](autodiff.md).

If a relaxed numerical mode is ever wanted, it must be an **explicit,
separately documented execution mode**. It is not introduced now.

---

## 9. Conformance testing requirements

**The specification is the oracle.** Expected values must be derived from this
document and written into the tests. A test must not compute its expectation
by running another backend, and must not treat the Python backend's output as
correct by construction.

**No such tests exist yet.** Nothing in this section has been executed.

### 9.1 Required coverage

| Area | What must be covered |
| --- | --- |
| Integer dtypes | `uint8`, `int8`, `int16`, `int32`, `int64`, each with `+`, `-`, `*` |
| Overflow boundaries | signed `max + 1`, `min - 1`, `min * -1`; unsigned `max + 1`, `0 - 1` |
| Wraparound | every row of [section 4.3](#43-worked-examples) |
| Construction and casting | out-of-range literal and `astype` still raise; **not** wrapped |
| Same-dtype promotion | every dtype with itself |
| Mixed-dtype promotion | all 49 cells of [section 6.2](#62-the-promotion-table), including the four `cast` cells raising |
| Promotion symmetry | `promote(a, b) == promote(b, a)` for all pairs |
| Scalar conversion | every row of [section 6.5](#65-python-scalars), including the raising cases |
| Scalar integer representability | integers on both sides of \(2^{p}\) for each floating dtype: \(2^{p}\), \(2^{p}+1\), \(2^{p+1}\), \(2^{p+1}+1\). A test asserting only the interval bound would wrongly reject \(2^{p+1}\) |
| Floating specials | `inf`, `-inf`, `nan`, `+0.0`, `-0.0`, and their propagation |
| Division by zero | the eight rows of [section 7.2](#72-division-by-zero); integer division by zero raising |
| Subnormals | subnormal operands **and** subnormal results, on every backend |
| Rounding boundaries | `1.0 + 2**-24`, `1.0 + 2**-23` in `float32`, and the binary64 equivalents |
| Signed zero | `0.0 + -0.0`, `-0.0 + -0.0`, `1.0 / -0.0` |
| Integer true division | every row of [section 7.3](#73-integer-true-division) |
| Cross-backend consistency | every case above, on every installed backend |
| Execution location | [section 9.4](#94-semantic-tests-versus-execution-tests) |
| Graph equivalence | fused and unfused execution of the same expression; replay after a backend switch |

### 9.2 Required strictness

- **Integer arithmetic: exact equality.** No tolerance, ever.
- **Floating arithmetic: the correctly rounded result**, compared **bitwise**,
  including the sign bit of zeros and infinities.
- **NaN: classified, not compared.** Assert that the result is NaN. Do not
  compare payloads or sign bits.
- **Result dtype**, asserted by identity (`assertIs(result.dtype, ts.int32)`).
- **Storage width**, so a `float32` result is not held as `float64`.

> **A numerical tolerance must not be used to conceal a backend's failure to
> implement a correctly rounded elementary operation.** For `+`, `-`, `*` and
> `/` the correct result is uniquely determined, so any tolerance would be
> hiding a defect. Tolerances belong to operations IEEE 754 does not require to
> be correctly rounded, which this document does not cover.

### 9.3 Cross-backend equality: what is achievable

Bitwise cross-backend equality is **required** for all four operations on both
floating dtypes, and for every integer operation.

It is also **achievable**, which was established rather than assumed. IEEE 754
requires these four operations to be correctly rounded, so the result is
uniquely determined; and measurement confirms no obstacle beyond the CuPy
default described in [section 5.4](#54-subnormals-gradual-underflow-is-required).

> **Correction to an earlier statement.** A previous revision of this document
> reported an experiment comparing NumPy on the CPU with CuPy on the GPU over
> ~394,000 random `float32` pairs per operation, and summarised it as showing
> agreement "for normal results". That summary was wrong, and the distinction
> matters.
>
> What the experiment established is agreement when **the operands were
> restricted to normal values and the result was also normal**. It did **not**
> establish agreement whenever only the result is normal: a subnormal
> *operand* flushed to zero on the device can produce a perfectly normal — and
> wrong — result. The `1e-40 * 1.0` case in
> [section 5.4](#54-subnormals-gradual-underflow-is-required) is of exactly
> that shape, and `2.1169182e-33 + -4.406691e-39` is an observed instance
> where both operands are finite, the result is normal on both sides, and the
> two disagree because the smaller operand was flushed.
>
> A conformance test must therefore **not** filter by result classification.
> It must cover subnormal operands explicitly, whatever the result looks like.

### 9.4 Semantic tests versus execution tests

These are different questions and belong in different tests:

- **Semantic conformance** — does the operation produce the specified value,
  dtype and exceptional behaviour? Runs on every backend; asserts against
  values from this document.
- **Execution location** — did the operation execute on the selected backend,
  and where does its result reside? Asserts about storage type and reported
  execution, not about numbers.

A semantic test that passes because the operation quietly fell back to another
backend is not evidence that the selected backend conforms. Keeping the two
separate is what makes that visible. The observability mechanism required for
the second is specified in [backends.md](backends.md).

### 9.5 The existing parity helper

`tests/backend/_support.py` defines `NumPyParityTestCase`, whose
`assertOperationParity` evaluates an expression on the Python backend and
asserts NumPy matches it. That is "Python is the reference" expressed as a test
helper.

It remains useful as a **cross-check** — two independent implementations
agreeing is evidence — but it must stop being the definition of correctness.
It must be re-cast so that both backends are compared against specified
values, not against each other.

---

## 10. Migration and compatibility

> Adopting this specification is a **deliberate set of breaking changes**, not
> a bug fix. Results, dtypes and raised exceptions all change.

### 10.1 Breaking changes

| # | Area | Current behaviour (verified) | Specified behaviour |
| --- | --- | --- | --- |
| B1 | Integer overflow | Raises `OverflowError` on every backend | Wraps silently: `int32` `2147483647 + 1` → `-2147483648` |
| B2 | Unsigned overflow | Raises | Wraps modulo \(2^8\) for `uint8` |
| B3 | Float division by zero | Raises `ZeroDivisionError` | IEEE result: `±inf`, or `nan` for `0/0` |
| B4 | Integer division by zero | Raises `ZeroDivisionError` | Unchanged — still raises |
| B5 | `int32 + float32` | `float64` (by `result_dtype`) | `float64` — same result, now by rule |
| B6 | `int64` with any float | Promotes to `float64`, losing precision above \(2^{53}\) | Raises; explicit cast required |
| B7 | `int64 / int64` | `float64` | Raises; explicit cast required |
| B8 | `int32_tensor + 3.5` | Promotes to `float64` | Raises `TypeError` |
| B9 | Scalar out of tensor range | Promotes to a wider dtype | Raises `TypeError` |
| B10 | `float32` working precision | Operands widened to `float64`, then narrowed | Computed at declared precision |
| B11 | `float32` overflow | NumPy and CUDA decline; falls back to Python | Produces `inf` natively |
| B12 | CUDA integer arithmetic | Rejected in `_operand`; always falls back to Python | Executes natively |
| B13 | CUDA `float32` subnormals | Flushed to zero | Gradual underflow required |
| B14 | Fused `a*b+c` on CUDA | May contract to an FMA (one rounding) | Must preserve both roundings |
| B15 | Explicit backend selection | May silently fall back to Python | Must execute or report unsupported |
| B16 | `tensor + True` | `bool` accepted as an integer scalar | Raises `TypeError` ([section 6.5](#65-python-scalars)) |

### 10.2 What may depend on the old behaviour

- **Code relying on `OverflowError`** to signal that an integer computation
  left its range will stop receiving it. Wraparound is silent by design.
- **Code relying on `ZeroDivisionError`** from floating division will now
  receive `inf` or `nan` and must check explicitly.
- **Code mixing `int64` with floats**, or dividing `int64`, now needs an
  explicit cast.
- **`float32` results** may change in the last ulp wherever an evaluation
  strategy previously carried a binary64 intermediate across more than one
  operation. Per single operation they will not change
  ([section 5.5.1](#551-a-single-widened-operation-is-not-observably-different)).
- **Benchmark baselines.** Measurements taken before this refactor are not
  comparable with measurements after it, because the conversions and fallbacks
  being removed are a large part of what the current numbers measure.

*Correction.* An earlier revision of this section claimed that no test
asserted `OverflowError` or `ZeroDivisionError`. The first half held; the
second did not. Four tests asserted `ZeroDivisionError` from floating
division — in `tests/tensor/test_ops.py`, `tests/backend/test_elementwise.py`
and `tests/backend/test_fusion.py` — and were rewritten against B3 when it was
implemented. The lesson is the one B16 repeats: the breaking-change list was
assembled by reading the implementation, and what the tests asserted was a
second source that should have been consulted too.

### 10.3 Implementation paths requiring review

**All of these have now been modified**, except where the last column says
otherwise. The table is kept as the record of what the refactor touched.

| Path | Why |
| --- | --- |
| `tensors/dtype.py` — `result_dtype`, `_integer_scalar_result_dtype`, `_INTEGER_LIMITS` | Promotion must be rewritten to [section 6.2](#62-the-promotion-table). The value-dependent scalar promotion contradicts [section 6.4](#64-what-is-not-promoted). |
| `tensors/backend/numpy/conversion.py` — `_operand`, `_storage` | Widens floats to `float64`; computes integers in `object` arrays; declines on narrowing overflow via a `bool(numpy.any(...))` check. |
| `tensors/backend/cuda/conversion.py` — `_operand`, `_storage` | Refuses integer dtypes; widens floats to `float64`; `bool(cupy.any(...))` forces a host synchronisation. |
| `tensors/backend/{python,numpy,cuda}/kernels/arithmetic/` | `add`, `subtract`, `multiply`, `divide` for all three backends. |
| `tensors/operations/arithmetic/divide.py` | Holds the operation-layer `ZeroDivisionError` guard (three sites). |
| `tensors/backend/policy.py` | Workload thresholds decide which path runs; under P3 the path must not change results, and under P4 it must not override explicit selection. The four arithmetic operations no longer consult it under any selection, automatic included; the operations outside this contract still do. |
| `tensors/backend/cuda/kernels/fusion/fused_elementwise.py`, `fused_elementwise_backward.py` | `cupy.RawKernel(source, name)` with no options permits FMA contraction (B14). Now compiled with `--fmad=false`, and the forward kernel no longer rejects a zero denominator. |
| `tensors/backend/config.py`, `types.py` | Backend selection; strict mode has no representation today. `"auto"` resolves once, to NumPy when available and Python otherwise, and is then indistinguishable from naming that backend. |
| `tests/backend/_support.py` — `NumPyParityTestCase` | Encodes Python-as-reference ([section 9.5](#95-the-existing-parity-helper)). **Not changed.** The conformance tests in `tests/operations/arithmetic/` compare against specified values instead, so the parity helper is no longer the only check; re-casting it is separate work. |
| `tensors/backend/{python,numpy,cuda}/kernels/elementwise/division_denominator_gradient.py` | The division VJP raised, or declined and let the Python reference raise, where the forward pass returns an infinity. Its zero-denominator and finiteness tests also read device memory back to the host on every backward pass. |
| `tensors/backend/dispatch/elementwise/`, `tensors/operations/_gradient_shaping.py` | **Not changed.** The `+`, `-` and `*` VJPs run through a fused multiply-and-reduce kernel, which is not one of the four operations, so its dispatch still applies the workload threshold. Measured: under explicit NumPy selection a 4-element backward pass returns `PythonStorage`; a 4096-element one returns `NumPyStorage`. Division is unaffected, its numerator gradient being a division. |
| `docs/backends.md`, `docs/autodiff.md` | Updated alongside this document. |

### 10.4 Implementation sequence

Reviewable stages, each leaving the tree working:

1. **Establish specification-derived conformance tests.** Written from this
   document, with expectations in the test. They will fail; that is the point.
2. **Implement dtype and integer arithmetic semantics.** Wraparound for all
   five integer dtypes; remove the `object`-array path; enable native CUDA
   integer kernels.
3. **Implement floating-point arithmetic semantics.** Declared precision; IEEE
   overflow; gradual underflow, including the inline-PTX `float32` path on
   CUDA ([section 5.4](#54-subnormals-gradual-underflow-is-required)).
4. **Implement promotion and division rules.** The table in
   [section 6.2](#62-the-promotion-table), scalar rule S, IEEE division by
   zero, integer true division rule D. Remove the host-synchronising zero
   check.
5. **Enforce backend execution requirements.** Strict versus automatic
   selection, the unsupported-operation error, and the observability mechanism
   ([backends.md](backends.md)).
6. **Verify graph and fusion equivalence.** Fused output must equal unfused
   output bitwise; compile fusion kernels so contraction cannot occur.
7. **Run the full test suite and the benchmarks.**

**Stage 7 must compare against the comprehensive benchmark baseline collected
before this refactor**, using the suite described in
[`benchmarks/README.md`](../benchmarks/README.md). Note that pre-refactor and
post-refactor numbers measure different code paths, so the comparison is for
detecting regressions and quantifying the change, not for claiming a
speed-up.

**No performance target is set here, and no improvement is claimed.** The
conversions and fallbacks being removed are expected to cost something, but
what they cost is a measurement, not a prediction.

---

## 11. Open specification questions

Each must be decided before the behaviour it governs is implemented. An
implementation agent must treat these as **do not touch**.

Every question that the previous revision left open because the implementation
behaved differently has now been decided: integer wraparound extends to all
five integer dtypes ([section 4](#4-integer-arithmetic)), division by zero is
resolved ([section 7.2](#72-division-by-zero)), integer true division is
resolved ([section 7.3](#73-integer-true-division)), promotion is resolved
([section 6](#6-result-dtypes-and-promotion)), and CUDA subnormal handling is
resolved as a requirement with an achievable implementation
([section 5.4](#54-subnormals-gradual-underflow-is-required)).

### O1. Floor division

What operation provides integer-valued division, what is its result dtype, how
does it round (toward zero, or toward negative infinity), what happens at
`min // -1`, and what happens on a zero denominator? A new public operator or
method would be an API addition and is out of scope here
([section 7.4](#74-floor-division-is-a-separate-operation)).

### O2. Remainder and divmod

Not currently public. If added, its sign convention must be specified together
with floor division, since the two are linked by the division identity.

### O3. Relaxed numerical mode

[Section 8.5](#85-graph-optimisation) forbids fusion from changing results. If
a faster, explicitly-opted-in mode permitting FMA contraction or reassociation
is wanted later, its selection mechanism, scope and documentation are
undecided. It is not introduced now.

### O4. IEEE exception flag exposure

[Section 7.2](#ieee-exception-flags-are-not-public-api) states that the flags
are not public API in this refactor. Whether to expose them later — as an
accessor, a context manager, or a trapping mode — is undecided.

### O5. Unsigned dtypes beyond `uint8`

`uint8` is the only public unsigned dtype. If `uint16`, `uint32` or `uint64`
were added, the promotion table would need new rows, and `uint64` with `int64`
would have no common public integer dtype — which P-e would turn into a
required cast.

---

## Related documents

- [Numerical backends](backends.md) — backend selection, strict and automatic
  execution, operation support, fallback, observability and storage residency.
- [Automatic differentiation](autodiff.md) — graph execution, replay, and the
  numerical equivalence required of fusion.
- [Tensor memory model](memory-model.md) — logical shape, strides, physical
  storage.
- [Package structure](package-structure.md) — where operations, dispatch and
  backend kernels live.
- [Benchmarks](../benchmarks/README.md) — the measurement suite and the
  baseline the refactor is compared against.
