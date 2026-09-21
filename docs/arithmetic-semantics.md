# Arithmetic semantics

## 1. Status and scope

> **Status: implemented for `+`, `-`, `*`, `/` and `**`; specification
> elsewhere.**
>
> Addition, subtraction, multiplication, true division and **exponentiation**
> follow this document on the Python, NumPy and CUDA backends, in eager
> execution, graph replay, differentiation and fusion.
>
> Exponentiation's milestones D1–D7 and S3 are implemented and reviewed:
> the IEEE `pow` function and its special-value table
> ([12.2](#122-the-function-d1), [12.3](#123-exceptional-values-d2)),
> fixed-width integer exponentiation and the rejection of negative integer
> exponents ([12.4](#124-integer-exponentiation)), result dtypes and scalar
> conversion including rule S3
> ([12.5](#125-result-dtypes-and-scalar-conversion),
> [6.5](#65-python-scalars)), the accuracy bounds and their validated
> reference ([12.6](#126-accuracy)), and the differentiation region table
> ([12.7](#127-differentiation-d7)). Forward power and both power-gradient
> **kernels** dispatch strictly: each executes on the selected backend or
> raises `BackendOperationUnsupportedError`. Subsequent gradient-shaping
> reductions are **not** yet guaranteed to execute on that backend; see the
> execution-location note below.
>
> **This document does not govern the rest of the package.** The other
> fifty-five public numerical operations — comparisons, `where`, `clip`,
> `concat`, reductions, matmul, the elementwise maths functions — have no
> numerical specification, and most still use the older promotion in
> `tensors/dtype.py` `result_dtype`. Extending a contract to them is separate
> work, surveyed in
> [the numerical-completeness audit](numerical-completeness-audit.md).
>
> Two promotion authorities therefore coexist, and they disagree on four of
> the forty-nine dtype cells: `int64` against either floating dtype, and its
> transposes. This document's [section 6.2](#62-the-promotion-table) requires
> a cast; `result_dtype` promotes silently to `float64` and loses `int64`
> values above \(2^{53}\). That conflict is **finding D-2 of the audit and is
> not resolved here.** Nothing in this document should be read as deciding
> which authority governs the operations outside its scope.
>
> **Reading the *Current behaviour* and *Historical behaviour* passages.**
> Both record what the package did *before* the work that this document
> specifies, kept because
> [section 10](#10-migration-and-compatibility) explains the migration away
> from it. **Neither describes current behaviour**, despite the older label,
> and neither may be cited as evidence that a behaviour exists today.
>
> Two gaps within the implemented scope are recorded rather than closed. The
> execution-location observability API of
> [backends.md](backends.md#observability) does not exist.
>
> The VJP execution-location requirement of [autodiff.md](autodiff.md) is
> **partly** met for power. Its two gradient *kernels* dispatch strictly, but
> the broadcast reduction that shapes their results still goes through
> `execute_sum_to_shape`, which applies the workload threshold, so a small
> power backward pass **with broadcasting** returns `PythonStorage` under
> explicit NumPy selection even though the gradient itself was computed by
> NumPy. Twenty-five of the twenty-seven gradient dispatchers remain on the
> legacy policy besides. See
> [section 10.3](#103-implementation-paths-requiring-review).

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

Elementwise **exponentiation** (`**`), specified in
[section 12](#12-exponentiation), including its result dtypes, integer and
floating-point semantics, accuracy, and differentiation. Sections 1–11 are
written for the four operations above; section 12 states where exponentiation
follows them and where it differs.

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
  preserve these semantics ([section 8.5](#85-graph-optimisation)) and the
  derivatives of exponentiation
  ([section 12.7](#127-differentiation-d7)).
- Transcendental operations other than `**`. The accuracy rules in
  [section 12.6](#126-accuracy) apply to exponentiation alone and are not
  extended to `exp`, `log` or the trigonometric functions by implication.

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

**P3. A supported operation is semantically identical regardless of backend,
tensor size or execution path.** An operation must not change its behaviour
because a tensor was small enough to take one path or large enough to take
another. Semantic identity covers the result dtype, the shape, every
exceptional value and its sign, NaN classification, and which conditions raise.

**P3 requires semantic consistency. It does not by itself require bitwise
reproducibility.** For an operation that IEEE 754 requires to be *correctly
rounded* — `+`, `-`, `*`, `/` — the result is uniquely determined, so semantic
identity and bitwise identity coincide, and
[section 9.3](#93-cross-backend-equality-what-is-achievable) requires the
latter. For an operation that IEEE 754 only *recommends*, and that no
mainstream implementation rounds correctly, bitwise identity is not achievable
and is not required; such an operation must instead state an accuracy bound
against the correctly rounded result. Exponentiation is the first such
operation, and states its bound in [section 12.6](#126-accuracy). **A bound
approved for one operation is never extended to another by implication.**

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

**Exponentiation uses this table unchanged**, with the base and the exponent as
the two operands, and adds a rule for the reflected form where the scalar is
the base. See [section 12.5](#125-result-dtypes-and-scalar-conversion). The
exponent is not exempted from promotion on the grounds that it plays a
different mathematical role: its value affects the result, so converting it to
a dtype that cannot represent it exactly would change the computation.

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

### 7.5 The division VJPs

For `y = a / b` with upstream gradient `g`, elementwise:

```
dy/da  ->   g / b
dy/db  ->  -g * a / b ** 2
```

**Execution.** Both follow [Execution requirements](backends.md#execution-requirements)
in full: the selection decides where they run, no workload-size policy
applies, there is no Python-reference fallback, and a declining kernel
raises `BackendOperationUnsupportedError`. The broadcast reductions that
shape each gradient back to its operand use the selected-backend reduction,
so a *broadcast* backward pass is as resident as a non-broadcast one — it
was not, before this was written.

**A zero denominator is a value, not an error.** The VJPs deliver
[section 7.2](#72-division-by-zero)'s result rather than raising:
`-g * a / b**2` at `b == 0` is the signed infinity the sign of `-g * a`
gives, and NaN when that product is zero. Nothing reads a denominator back
to the host to discover this, so there is no synchronisation on a device.
The graph-built VJP agrees with the eager one here; it used to raise
`ZeroDivisionError` because it scanned the denominator's host values once,
when the operation was recorded.

**Range safety.** `-g * a / b ** 2` is not evaluated as written when
`b ** 2` would overflow or underflow although the quotient is
representable. The kernels select a logarithmic form for exactly those
elements, decided elementwise on the device. `1e300 / 1e200` differentiates
to `-1e-100`, where forming `b ** 2` first gives zero.

**Higher order.** Writing `r = -g * a / b**2`, the second-order partials are

```
dr/dg  =  -outer * a / b ** 2
dr/da  =  -outer * g / b ** 2
dr/db  =   2 * outer * g * a / b ** 3
```

The first two reuse the range-safe primitive directly. The third is
factored through it as `-2 * (a / b) * (-outer * g / b**2)` rather than
forming `b ** 3`, which keeps the cubed denominator from overflowing on its
own. That is a change of floating-point behaviour from the exact-rational
host loop it replaces — the loop evaluated the whole product as one
rational — and it is the price of the partials executing where the
selection says instead of in Python. The eager and graph-built paths now
agree, which they did not while one was exact-rational and the other was
ordinary arithmetic.

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

**Exponentiation divides by dtype**, and the exclusion must not be read more
broadly than it is:

- **Integer exponentiation is exact fixed-width arithmetic**
  ([section 12.4.1](#1241-non-negative-exponents-wrap-d3)) and is therefore
  **required to be bitwise identical** across conforming backends, exactly as
  every other integer operation is.
- **Floating-point exponentiation** requires the IEEE special values of
  [section 12.3.3](#1233-the-complete-special-value-table) to agree
  **exactly**, including the prescribed signs of zeros and infinities, with
  NaN compared by classification. Its **ordinary finite results are bounded,
  not bitwise**: they must satisfy the accuracy limits of
  [section 12.6.2](#1262-the-accuracy-bounds) against the correctly rounded
  value.

This is the distinction the amended **P3** in
[section 2](#2-architectural-principles) draws between semantic consistency and
bitwise reproducibility.

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

Exponentiation adds the following. All are specified in
[section 12](#12-exponentiation) and **none is implemented**.

| # | Area | Current behaviour (verified) | Specified behaviour |
| --- | --- | --- | --- |
| B17 | `(-2.0) ** 0.5` | Raises `ValueError` | `nan` ([12.3.1](#1231-the-invalid-class)) |
| B18 | `0.0 ** -1.0` | Raises `ValueError` | `+inf`; `-0.0 ** -1.0` is `-inf` ([12.3.2](#1232-the-divide-by-zero-class)) |
| B19 | `float64` power overflow | Raises `OverflowError` | `+inf` ([12.3.4](#1234-overflow-and-underflow)) |
| B20 | Integer power overflow | Raises `OverflowError` | Wraps at the promoted width ([12.4.1](#1241-non-negative-exponents-wrap-d3)) |
| B21 | `int32 ** -1` | `float64` `[0.5]` | Raises `ValueError` ([12.4.2](#1242-negative-exponents-raise-d4)) |
| B22 | Power result dtype | Depends on exponent **values** | Static, from declared dtypes ([12.5](#125-result-dtypes-and-scalar-conversion)) |
| B23 | `int32 ** 2.0` | `float64` | `int32` by S2 ([12.5.2](#1252-tensor-base-with-a-python-scalar-exponent)) |
| B24 | `int32 ** 0.5` | `float64` | Raises `TypeError` by S2 ([12.5.2](#1252-tensor-base-with-a-python-scalar-exponent)) |
| B25 | `int64` with a floating operand in `**` | `float64`, losing precision above \(2^{53}\) | Raises; explicit cast required ([12.5.1](#1251-tensor-base-with-tensor-exponent)) |
| B26 | `-2 ** uint8_tensor` | `int16` — a scalar widened the result | Raises `TypeError` by S1 ([12.5.3](#1253-python-scalar-base-with-tensor-exponent)) |
| B27 | Negative base, tensor exponent, backward | Raises, discarding a valid base gradient | Base gradient returned; exponent gradient `nan` ([12.7.2](#1272-the-region-table)) |
| B28 | Zero-base gradients | Raise | Specified per region, `nan` or `+inf` ([12.7.2](#1272-the-region-table)) |
| B29 | Cross-backend `**` results | Believed identical (an artefact of the Python fallback) | Bounded, not bitwise ([12.6](#126-accuracy)) |

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

Applying that lesson to exponentiation, the tests were read before B17–B29 were
written. `tests/backend/test_power_execution.py` asserts `ValueError` and
`OverflowError` from power's domain and overflow cases, and
`tests/tensor/test_power.py` asserts the current dtype and zero-base behaviour;
both must be rewritten when [section 12](#12-exponentiation) is implemented.
The `OverflowError` assertions in `tests/operations/arithmetic/` are a
different matter: they cover **construction and casting**, which
[section 4.5](#45-arithmetic-is-not-construction-or-casting) keeps raising, and
are unaffected.

### 10.3 Implementation paths requiring review

**All of these have now been modified**, except where the entry says
otherwise. The table is kept as the record of what the refactor touched.

The *Why* column states the **reason each path was listed** — that is, what it
did *before* it was changed. It is not a description of current behaviour.
Entries re-audited against the source at the Phase 0 reconciliation say so
explicitly and give the outcome.

| Path | Why |
| --- | --- |
| `tensors/dtype.py` — `result_dtype`, `_integer_scalar_result_dtype`, `_INTEGER_LIMITS` | Promotion must be rewritten to [section 6.2](#62-the-promotion-table). The value-dependent scalar promotion contradicts [section 6.4](#64-what-is-not-promoted). |
| `tensors/backend/numpy/conversion.py` — `_operand`, `_storage` | Widens floats to `float64`; computes integers in `object` arrays; declines on narrowing overflow via a `bool(numpy.any(...))` check. |
| `tensors/backend/cuda/conversion.py` — `_operand`, `_storage` | Refuses integer dtypes; widens floats to `float64`; `bool(cupy.any(...))` forces a host synchronisation. |
| `tensors/backend/{python,numpy,cuda}/kernels/arithmetic/` | `add`, `subtract`, `multiply`, `divide` for all three backends. |
| `tensors/operations/arithmetic/divide.py` | Holds the operation-layer `ZeroDivisionError` guard (three sites). |
| `tensors/backend/policy.py` | Workload thresholds decide which path runs; under P3 the path must not change results, and under P4 it must not override explicit selection. **Re-audited at Phase 0:** the five arithmetic operations and power's two gradients no longer consult it under any selection, automatic included. The operations outside this contract still do — 108 of the 121 dispatch modules. |
| `tensors/backend/cuda/kernels/fusion/fused_elementwise.py`, `fused_elementwise_backward.py` | `cupy.RawKernel(source, name)` with no options permits FMA contraction (B14). Now compiled with `--fmad=false`, and the forward kernel no longer rejects a zero denominator. |
| `tensors/backend/config.py`, `types.py` | Backend selection; strict mode has no representation today. `"auto"` resolves once, to NumPy when available and Python otherwise, and is then indistinguishable from naming that backend. |
| `tests/backend/_support.py` — `NumPyParityTestCase` | Encodes Python-as-reference ([section 9.5](#95-the-existing-parity-helper)). **Not changed.** The conformance tests in `tests/operations/arithmetic/` compare against specified values instead, so the parity helper is no longer the only check; re-casting it is separate work. |
| `tensors/backend/{python,numpy,cuda}/kernels/elementwise/division_denominator_gradient.py` | The division VJP raised, or declined and let the Python reference raise, where the forward pass returns an infinity. Its zero-denominator and finiteness tests also read device memory back to the host on every backward pass. **Re-audited at Phase 0:** the numerical behaviour was corrected with the division work and the finiteness read is gone, but its *dispatcher* still applied the workload threshold and fell back to the Python reference. **Closed:** the dispatcher is now strict, the three kernels take prepared native operands, and both broadcast reductions use the selected-backend reduction. See [section 7.5](#75-the-division-vjps). |
| `tensors/backend/dispatch/reductions/`, `tensors/backend/dispatch/elementwise/`, `tensors/operations/_gradient_shaping.py` | The `+`, `-` and `*` VJPs reduced and negated through entry points that applied the workload threshold, so a small backward pass under explicit NumPy returned `PythonStorage`. They now dispatch through `execute_vjp_sum_to_shape`, `execute_vjp_negate` and `execute_sum_products_to_shape`, which honour the selection at every size. **Re-audited at Phase 0:** power's two gradient dispatchers are strict as well, so **2 of the 27 gradient dispatchers** meet the execution-location requirement. The other 25 — including the division VJP, which this table lists separately — still apply the workload threshold and fall back to the Python reference. Power is not fully closed either: `sum_to_shape` in `_gradient_shaping.py` still uses `execute_sum_to_shape`, so a broadcast power backward pass under explicit NumPy returns `PythonStorage` below the threshold while the same pass without broadcasting returns `NumPyStorage`. Extending strict dispatch past the arithmetic four is separate work. |
| `tensors/operations/arithmetic/power.py` — `_power_dtype`, `_scalar_base_power_dtype` | Value-dependent result dtype, contradicting [section 6.4](#64-what-is-not-promoted); read `exponent._data`, a host transfer. **Done (D6).** Both functions are gone; `resolve_power` and `resolve_power_scalar_base` in `tensors/dtype.py` resolve from declared dtypes alone. |
| `tensors/backend/python/kernels/elementwise/power_base_gradient.py` | Contained a **second copy** of `_power_dtype` with the same defect. **Done (D6).** No copy remains anywhere in `tensors/`. |
| `tensors/operations/arithmetic/power.py` — `Pow.backward`, `Pow.backward_graph` | Value-reading domain checks that raised and discarded valid gradients (B27, B28). **Done (D7).** Both now apply the region table of [12.7.2](#1272-the-region-table); neither reads an operand. |
| `tensors/backend/{python,numpy,cuda}/kernels/arithmetic/power.py` | Domain and overflow handling (B17–B20); the array kernels declined where the reference raised. **Done (D1, D2).** No kernel declines for a numerical condition, on any dtype or operand form. |
| `tensors/backend/dispatch/elementwise/power_base_gradient.py`, `power_exponent_gradient.py` | Workload threshold and reference fallback; the exponent kernel also declined through `bool(numpy.any(bases <= 0.0))`, a host synchronisation. **Done (D7, G6).** Both are strict: no threshold, no fallback, no host read, and a decline raises `BackendOperationUnsupportedError`. |
| `tensors/backend/cuda/kernels/fusion/errors.py` | Power domain codes 8, 9 and 14 replicated the raising behaviour; they had to go when [12.3](#123-exceptional-values-d2) and [12.7](#127-differentiation-d7) landed, or fused and eager execution would diverge. **Done.** All three codes are removed and power has neither a forward nor a backward domain check. |
| `tensors/variable.py` — `__pow__` | Called `_power_dtype` with a `Variable` exponent, which crashed for integer dtypes. **Done (D6).** `__pow__` and `__rpow__` use `resolve_power` and `resolve_power_scalar_base`. |
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

Exponentiation's seven numerical decisions — the choice of IEEE `pow`,
non-trapping exceptional values, integer wraparound, negative integer
exponents, accuracy, promotion and scalar conversion, and differentiation —
were each resolved individually and are recorded in
[section 12](#12-exponentiation). None of them is open.

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

---

## 12. Exponentiation

> **Status: implemented and reviewed.** Milestones D1–D7 and S3 are complete
> on the Python, NumPy and CUDA backends, in eager execution, graph replay,
> differentiation and fusion, and forward power and both power gradients
> execute on the selected backend or raise.
>
> Passages below marked *Historical behaviour* record what the package did
> **before** that work. They are retained because
> [section 10](#10-migration-and-compatibility) explains the migration, and
> each states the behaviour that replaced it. **None of them describes the
> package as it is now.**

This section is placed after section 11 so that no existing section is
renumbered and no cross-reference in this document breaks. It is not an
appendix: it carries the same authority as the sections before it, and it
extends them rather than competing with them. Where a rule already exists —
the promotion table, the scalar rules, wraparound, rounding — exponentiation
uses that rule and says so, rather than restating it.

Seven decisions were resolved individually before this section was written.
They are referenced below as **D1** to **D7**.

### 12.1 Scope

`**` on tensors, in all three forms: `tensor ** tensor`, `tensor ** scalar`
and `scalar ** tensor` (reflected). Covers `Tensor.__pow__`, `Tensor.__rpow__`,
`Variable.__pow__`, `Variable.__rpow__` and the `pow` function, in eager
execution, graph replay and fusion, together with differentiation of the
result.

Out of scope: floor division ([section 7.4](#74-floor-division-is-a-separate-operation)),
broadcasting, and every other transcendental operation. **The accuracy rules in
[section 12.6](#126-accuracy) apply to `**` alone** and are not extended to
`exp`, `log`, the trigonometric functions or any other operation by implication.

### 12.2 The function (D1)

**Floating-point exponentiation is IEEE 754-2019 clause 9.2 `pow`.**

IEEE 754 describes three distinct power functions. The choice among them is a
semantic decision, not a detail:

| function | behaviour | chosen |
| --- | --- | --- |
| `pow(x, y)` | C99-style; `pow(1, NaN) = 1`, `pow(NaN, 0) = 1`; negative bases permitted with integral exponents | **yes** |
| `powr(x, y)` | defined as \(e^{y \ln x}\); NaN for every \(x < 0\); `powr(±0, ±0)` is NaN | no |
| `pown(x, n)` | integer exponents only | no |

Consequences of choosing `pow`:

- **Results are strictly real.** `(-8.0) ** (1/3)` is NaN, not a complex cube
  root. Complex numbers are not part of this package and are not introduced by
  exponentiation.
- **A negative base with an integral exponent is well defined**:
  `(-2.0) ** 3.0` is `-8.0`, and `(-2.0) ** -3.0` is `-0.125`.
- Integer exponentiation is specified separately in
  [section 12.4](#124-integer-exponentiation); D1 governs floating-point
  operands only.
- Choosing `pow` does **not** imply bitwise-identical results across backends.
  Reproducibility is settled in [section 12.6](#126-accuracy).

*Historical behaviour (before D1, verified at the time):* the Python
reference called `math.pow`, which implements `pow` semantics but converts
IEEE signals into Python exceptions; NumPy and CuPy already delivered the IEEE
results. **Superseded.** The Python kernel now implements the non-trapping
clause 9.2 function directly, in the order the standard states its rows, and
`(-2.0) ** 0.5` returns NaN on all three backends.

### 12.3 Exceptional values (D2)

**Floating-point exponentiation does not raise on a numerical condition. It
delivers the IEEE result.** This is the same decision already taken for
division in [section 7.2](#72-division-by-zero), and for the same reason: where
the declared dtype holds a value that expresses the outcome, that value is the
result.

**This section governs floating-point operands only.** Integer exponentiation
has no infinity and no NaN to deliver, and
[section 12.4.2](#1242-negative-exponents-raise-d4) requires it to raise on a
negative exponent — the same division of responsibility as
[section 7.2](#72-division-by-zero), where floating division delivers an
infinity and integer division raises.

**A backend must not introduce a host synchronisation to detect any condition
in this section.** The IEEE result is produced by the hardware; there is
nothing to check.

#### 12.3.1 The invalid class

IEEE `pow` has exactly one invalid case:

> finite \(x < 0\) with finite non-integral \(y\) → **NaN**

`(-2.0) ** 0.5` is NaN. `(-8.0) ** (1/3)` is NaN. Neither raises.

#### 12.3.2 The divide-by-zero class

> \(\pm 0\) raised to a negative exponent → an infinity, with the sign the
> standard prescribes

| expression | result |
| --- | --- |
| `0.0 ** -1.0` | `+inf` |
| `-0.0 ** -1.0` | `-inf` |
| `-0.0 ** -2.0` | `+inf` |
| `-0.0 ** -2.5` | `+inf` |

The sign is negative only when the base is `-0.0` **and** the exponent is an
odd integer.

#### 12.3.3 The complete special-value table

Every row is **exact** on every backend. These are not approximations and are
not covered by the accuracy bounds of [section 12.6](#126-accuracy).

| expression | result | note |
| --- | --- | --- |
| `x ** ±0.0` | `1.0` | for every `x`, including NaN and infinities |
| `1.0 ** y` | `1.0` | for every `y`, including NaN |
| `-1.0 ** ±inf` | `1.0` | |
| `±0.0 ** y`, `y > 0` odd integer | `±0.0` | sign preserved |
| `±0.0 ** y`, `y > 0` otherwise | `+0.0` | |
| `±0.0 ** y`, `y < 0` odd integer | `±inf` | divide-by-zero |
| `±0.0 ** y`, `y < 0` otherwise | `+inf` | divide-by-zero |
| `x ** +inf`, \(\lvert x \rvert < 1\) | `+0.0` | |
| `x ** +inf`, \(\lvert x \rvert > 1\) | `+inf` | |
| `x ** -inf`, \(\lvert x \rvert < 1\) | `+inf` | |
| `x ** -inf`, \(\lvert x \rvert > 1\) | `+0.0` | |
| `-inf ** y`, `y > 0` odd integer | `-inf` | |
| `-inf ** y`, `y > 0` otherwise | `+inf` | |
| `-inf ** y`, `y < 0` odd integer | `-0.0` | |
| `-inf ** y`, `y < 0` otherwise | `+0.0` | |
| `+inf ** y`, `y > 0` | `+inf` | |
| `+inf ** y`, `y < 0` | `+0.0` | |
| finite `x < 0`, finite non-integral `y` | `NaN` | invalid |
| `NaN ** y`, `y ≠ 0` | `NaN` | |
| `x ** NaN`, `x ≠ 1` | `NaN` | |

#### 12.3.4 Overflow and underflow

Overflow produces `±inf`; underflow is gradual and then `±0.0`. Neither
raises, at either precision. `1e200 ** 2.0` is `+inf`.

*Historical behaviour (before D2, verified at the time):* `(-2.0) ** 0.5`
raised `ValueError`, `0.0 ** -1.0` raised `ValueError`, and `1e200 ** 2.0`
raised `OverflowError`. The exceptions came from CPython's `math.pow`, not
from a considered position; the array kernels declined in these cases and the
Python reference then raised. `float32` overflow returned `inf` while
`float64` overflow raised, because the reference computed in binary64 and
narrowed afterwards. **Superseded.** All three are now values on every
backend and at both precisions — `nan`, `inf` and `inf` respectively — and no
kernel declines for them.

#### 12.3.5 IEEE exception flags

As in [section 7.2](#72-division-by-zero), the sticky flags — *invalid*,
*divideByZero*, *overflow*, *underflow*, *inexact* — are **not** public API.
There is no accessor and no trapping mode. A caller who needs to know whether a
result is NaN tests the result.

### 12.4 Integer exponentiation

Integer exponentiation is not IEEE `pow`. It is exact integer arithmetic in
the declared width, and it follows [section 4](#4-integer-arithmetic).

#### 12.4.1 Non-negative exponents wrap (D3)

For a declared integer dtype of width \(w\) and an exponent \(n \geq 0\):

$$\operatorname{pow}_w(x, n) = \operatorname{wrap}_w\!\left(x^{n}\right)$$

where \(\operatorname{wrap}_w\) is the rule in
[section 4.2](#42-the-wraparound-rule). The result keeps the integer dtype.
**Overflow neither raises nor promotes.**

| expression | result |
| --- | --- |
| `int32: 2 ** 31` | `-2147483648` |
| `uint8: 16 ** 2` | `0` |
| `int8: (-2) ** 7` | `-128` |
| `int32: 2 ** 3` | `8` |
| `int32: 0 ** 0` | `1` |

The width is the **promoted result width** from
[section 12.5](#125-result-dtypes-and-scalar-conversion), not the base's width.
`uint8 ** int8` promotes to `int16`, so wraparound is at 16 bits.

**Implementation requirement.** The result must be computed by modular
exponentiation in the declared width. Materialising \(x^{n}\) as an
arbitrary-precision Python integer and truncating afterwards is not acceptable:
`int64: 3 ** 1000000` would build a 1.5-million-bit integer to return one
64-bit value.

#### 12.4.2 Negative exponents raise (D4)

> **Integer exponentiation requires a non-negative exponent. If any exponent
> element is negative, the operation raises `ValueError`.**

The rule is uniform. It applies whatever the base, including the bases whose
reciprocals happen to be representable:

| expression | result |
| --- | --- |
| `int32([2]) ** -1` | `ValueError` |
| `int32([1]) ** -1` | `ValueError` |
| `int32([-1]) ** -1` | `ValueError` |
| `int32([2]) ** int32([-1])` | `ValueError` |

`ValueError`, not `DtypePromotionError`: the operand dtypes are compatible and
the promotion table is satisfied. It is the exponent's **value** that leaves
the integer exponentiation domain.

The rationale is the one already given for integer division by zero in
[section 7.2](#72-division-by-zero): there is no integer value to deliver.
`2 ** -1` is \(1/2\), which no integer dtype represents, and truncating it to
`0` would discard the result rather than report it.

A caller who wants a fractional result casts first:
`t.astype(ts.float64) ** -1`.

**This does not weaken [section 6.4](#64-what-is-not-promoted).** The result
*dtype* never depends on values; it is fixed by
[section 12.5](#125-result-dtypes-and-scalar-conversion) before any element is
examined. D4 is a domain condition on the operation, evaluated after the dtype
is settled — exactly as a zero denominator is for integer division.

**Detection.** An accelerated backend may use a device-side reduction to
detect a negative exponent. It must not transfer the exponent tensor to the
host, and it must not fall back to another backend.

*Historical behaviour (before D4 and D6, verified at the time):*
`int32([2]) ** -1` returned `float64([0.5])`, and the result dtype was chosen
by inspecting exponent values — `int32 ** int32` yielded `int32` for exponents
`[1, 2]` and `float64` for `[1, -2]`. **Superseded.** A negative integer
exponent now raises `ValueError`, and the result dtype follows the declared
dtypes alone: `int32 ** int32` is `int32` whatever the exponent values, so
`[1, 2]` gives `int32` and `[1, -2]` raises rather than changing the dtype.

### 12.5 Result dtypes and scalar conversion

**Exponentiation introduces no new promotion table.**
[Section 6.2](#62-the-promotion-table) and the scalar rules S1–S4 in
[section 6.5](#65-python-scalars) govern it, with one addition: a rule for the
reflected form, where the scalar is the *base* and the tensor is the
*exponent*.

The result dtype is a static property of the declared operand dtypes and the
Python type of any scalar. **It never depends on element values**, as
[section 6.4](#64-what-is-not-promoted) requires.

#### 12.5.1 Tensor base with tensor exponent

The result dtype is [section 6.2](#62-the-promotion-table) applied to the two
declared dtypes, unchanged. `cast` cells raise and require an explicit cast.

The exponent participates in promotion exactly as any other operand. It is not
exempted on the grounds that it plays a different mathematical role: **the
exponent's numerical value affects the result**, and converting an integer
exponent to a floating dtype that cannot represent it exactly would change the
computation.

| expression | result dtype |
| --- | --- |
| `int32 ** int32` | `int32` |
| `uint8 ** int8` | `int16` |
| `int32 ** float32` | `float64` |
| `float32 ** float32` | `float32` |
| `float32 ** float64` | `float64` |
| `float32 ** int64` | **`cast` — raises** |
| `float64 ** int64` | **`cast` — raises** |
| `int64 ** float32` | **`cast` — raises** |
| `int64 ** float64` | **`cast` — raises** |

#### 12.5.2 Tensor base with a Python scalar exponent

S1–S4 apply with the tensor's dtype as the conversion target. The result has
the tensor's dtype. **A scalar never widens it.**

| expression | result | rule |
| --- | --- | --- |
| `int32_t ** 2` | `int32` | S1 |
| `int32_t ** 2.0` | `int32` | S2 — integral and in range |
| `int32_t ** 0.5` | **`TypeError`** | S2 — a non-integral float never converts to an integer dtype |
| `int32_t ** -1` | **`ValueError`** | dtype `int32` by S1; the value is rejected by [12.4.2](#1242-negative-exponents-raise-d4) |
| `uint8_t ** 300` | **`TypeError`** | S1 — 300 is outside `uint8` |
| `float32_t ** 2` | `float32` | S4 — 2 is exactly representable |
| `float32_t ** 0.5` | `float32` | S3 |

#### 12.5.3 Python scalar base with tensor exponent

The scalar base is weakly typed, but S1–S4 cannot be applied literally here:
they assume both operands play the same role. Forcing a *base* into the
*exponent's* dtype would make `2.5 ** int32_t` raise, refusing an ordinary real
result because the exponent happens to be an integer tensor.

> **S-p — a Python scalar base with a tensor exponent.**
>
> - Exponent tensor is **floating**: the target is that floating dtype; S3
>   governs a Python `float` base and S4 a Python `int` base.
> - Exponent tensor is **integer** and the base is a Python `int`: the target
>   is the exponent's integer dtype under S1, and the result has that dtype.
> - Exponent tensor is **integer** and the base is a Python `float`: the base
>   is taken as `float64`, the default floating dtype, and the promotion
>   restrictions of [section 6.2](#62-the-promotion-table) then apply between
>   `float64` and the exponent's dtype.

| expression | result | rule |
| --- | --- | --- |
| `2 ** int32_t` | `int32` | S-p, S1 |
| `-2 ** uint8_t` | **`TypeError`** | S-p, S1 — −2 is outside `uint8` |
| `2.5 ** int32_t` | `float64` | S-p — `float64` with `int32` promotes to `float64` |
| `2.5 ** int64_t` | **`cast` — raises** | S-p — `float64` with `int64` is a `cast` cell |
| `2 ** float32_t` | `float32` | S-p, S4 |
| `2.5 ** float32_t` | `float32` | S-p, S3 |

**Reflected exponentiation does not bypass the precision protections that
apply to the tensor–tensor form.** `2.5 ** int64_t` raises for the same reason
`float64 ** int64` raises.

*Historical behaviour (before D6 and S3, verified at the time):*
`int32 ** 2.0` gave `float64`; `int32 ** 0.5` gave `float64`;
`(-2) ** uint8_t` gave `int16`, a scalar widening the result; and the four
`int64` `cast` cells silently gave `float64`, losing precision above
\(2^{53}\). **Superseded.** `int32 ** 2.0` is now `int32`, because rule S3
converts the scalar to the base's dtype rather than widening the result;
`int32 ** 0.5` raises `TypeError`, because `0.5` has no `int32` value;
`(-2) ** uint8_t` raises `TypeError` under S1, because `-2` is outside
`uint8`; and the four `int64` `cast` cells raise `DtypePromotionError` rather
than losing precision.

Note that `-2 ** uint8_t` still evaluates to `int16`: Python parses it as
`-(2 ** uint8_t)`, so `2 ** uint8_t` is `uint8` under S1 and the widening
comes from the negation that follows, in `tensors/dtype.py` `negation_dtype`.
Unary negation is not specified by this document.

#### 12.5.4 Dtype resolution is not numerical evaluation

The result dtype is settled first, from declarations alone. Wraparound
([12.4.1](#1241-non-negative-exponents-wrap-d3)), the negative-exponent
rejection ([12.4.2](#1242-negative-exponents-raise-d4)) and the IEEE special
cases ([12.3](#123-exceptional-values-d2)) are then evaluated in that dtype.
The two stages never interact.

### 12.6 Accuracy

IEEE 754 requires correct rounding for `+`, `-`, `*`, `/` and `sqrt`, which is
why [section 9.3](#93-cross-backend-equality-what-is-achievable) can require
bitwise cross-backend equality for them: the result is uniquely determined.
**`pow` is in clause 9.2, among the recommended operations, and no mainstream
implementation rounds it correctly.** That justification does not transfer, and
this section states what does apply instead.

#### 12.6.1 Special values are exact

Every row of [section 12.3.3](#1233-the-complete-special-value-table) must be
reproduced **exactly** on every backend, including the prescribed signs of
zeros and infinities. NaN is compared **by classification**; its payload and
sign are unspecified and must not be tested.

#### 12.6.2 The accuracy bounds

> **Every other result must be within 2 ULP (`float64`) or 4 ULP (`float32`)
> of the correctly rounded result, per element, in the declared result dtype,
> on every backend.**

Per element, not on average: a single element outside the bound is a
conformance failure.

**Integer exponentiation is exact** and is not bounded: it is integer
arithmetic in the declared width under
[section 12.4.1](#1241-non-negative-exponents-wrap-d3), and every backend must
produce the identical integer. Bitwise cross-backend equality **is** required
there, as [section 9.3](#93-cross-backend-equality-what-is-achievable) requires
of every integer operation.

#### 12.6.3 These bounds are inherited, not proven

The figures are the loosest published by the implementations this package
depends on — NVIDIA documents a maximum of 2 ULP for `pow` and 4 ULP for
`powf`. **They are not mathematical guarantees, and this package does not
present them as such:**

- NVIDIA states its bounds are *"derived from extensive, though not
  exhaustive, testing. Therefore, they are not guaranteed."*
- glibc lists *"the maximum error … exposed by one of the existing tests"* and
  states that it *"does not aim for correctly rounded results"*.
- Microsoft's UCRT documents no per-function figure, saying only that results
  are *"in most cases … within ±1 ULP"*.

A future library, driver or toolkit update **can** violate these bounds while
remaining within its own vendor's documented behaviour. The bounds are
therefore **MS-Tensors conformance requirements that are tested**, not
properties inherited from a dependency.

#### 12.6.4 The ULP metric

Map a finite value to a monotone integer: interpret its bit pattern as a signed
integer \(i\); if \(i < 0\), replace it with \(\mathrm{INT\_MIN} - i\). Adjacent
representable values then differ by exactly 1, including across the
subnormal-to-normal boundary. **The ULP distance is the absolute difference of
these integers.**

The metric applies **only** to finite, non-zero results of the same sign.
Everything else is governed by [12.6.1](#1261-special-values-are-exact):

- **±0** — compared exactly, sign included. The integer mapping cannot express
  signed zero, which is why zero is an exact case rather than a bounded one.
- **±inf** — compared exactly. A finite result where the reference is infinite,
  or the reverse, is a failure, not a large distance.
- **NaN** — compared by classification only.
- **Opposite signs** between two non-zero finite values is a failure, not a
  distance.

#### 12.6.5 Measurement

Accuracy is measured against a **validated high-precision reference**, never
against another backend. The reference must establish a **valid error
enclosure** before a result may be called correctly rounded: it is not enough
to compute at some fixed precision and assume the answer.

A conforming procedure:

1. For an integral exponent, evaluate \(x^{n}\) exactly in rational arithmetic.
2. Otherwise evaluate \(e^{y \ln x}\) at working precision \(p\), carrying a
   rigorous bound on the accumulated relative error, including the
   amplification \(\lvert y \ln x \rvert\) introduced by the exponential.
3. Round **both** ends of the resulting enclosure into the target format,
   directly from the rational value, so that `float32` is rounded once rather
   than through `float64`.
4. Accept the result only when both ends round to the same value. Otherwise
   increase \(p\) and repeat.
5. If the enclosure has not resolved at the largest supported precision,
   **fail**. Never return an unvalidated reference.

**No fixed precision is provably sufficient.** A true value arbitrarily close
to a rounding boundary needs arbitrarily more precision to resolve — the
table-maker's dilemma — and the worst cases of `pow` are not catalogued for any
of the libraries involved. Step 4 is what makes the procedure sound; a fixed
60-digit evaluation is not.

Python's `decimal.ln()` and `decimal.exp()` are documented as correctly
rounded and are suitable for step 2. `decimal`'s own power operator is
documented as only *"almost always correctly rounded"* and **must not** be used
as the reference.

#### 12.6.6 Reproducibility

**Bitwise equality across backends is not required for `**`.** Two conforming
backends may differ by up to twice the bound in
[12.6.2](#1262-the-accuracy-bounds). Code that requires bit-identical results
across devices must not use `**`.

Semantic consistency is still required in full: the special-value table, the
signs of zeros and infinities, NaN classification, the result dtype, and every
rule in this section hold identically on every backend. See the amended **P3**
in [section 2](#2-architectural-principles).

#### 12.6.7 Determinism

Two requirements are stated here. They are independent, and neither implies
the other.

**R1 — Every supported execution path must satisfy this section's numerical
contract.** Whatever backend runs the operation, whatever the tensor's size or
memory layout, whether the work is fused or unfused, and whatever launch
configuration a kernel chooses, the result must satisfy the accuracy bounds of
[12.6.2](#1262-the-accuracy-bounds), reproduce the special values of
[12.3.3](#1233-the-complete-special-value-table) exactly, and carry the dtype
that [12.5](#125-result-dtypes-and-scalar-conversion) determines. This is a
requirement on every path, unconditionally. It does not promise that two paths
agree bit for bit.

**R2 — Bitwise determinism is promised only within a fixed execution
environment.** Repeating an operation returns identical bits when **all** of
the following are held fixed:

| condition | must be identical |
| --- | --- |
| backend | the same backend implementation |
| libraries | the same NumPy, CuPy, math-library, driver and toolkit builds |
| device | the same CPU model, or the same GPU model and compute capability |
| dtype | the same declared operand and result dtypes |
| execution configuration | the same execution path, memory layout, tensor shape and kernel launch configuration |

Change any of them and bitwise determinism is no longer promised, only R1.

**Bitwise determinism must not be inferred from measurement.** Observing that
two memory layouts, two launch configurations or two execution paths currently
agree does not establish that they must, and no such observation is promoted
to a guarantee here. Where the conditions above are not all fixed, R1 is the
only commitment.

Determinism is in particular **not** promised across backends, across library,
driver or toolkit versions, across devices, or across CPU models. Microsoft
documents that the UCRT selects implementations at run time and *"may produce
different results across CPUs"*, so two hosts running the same build may
legitimately differ while both conform.

### 12.7 Differentiation (D7)

This section governs the derivatives of `**`.
[autodiff.md](autodiff.md) governs where they execute and how graph
optimisation must preserve them.

For \(f(x, y) = x^{y}\), the partial derivatives are

$$\frac{\partial f}{\partial x} = y\,x^{y-1}, \qquad
  \frac{\partial f}{\partial y} = x^{y}\ln x$$

#### 12.7.1 Rules

> **G1 — The two gradients are independent.** Each is computed and returned on
> its own. A condition affecting one never suppresses the other. Only a
> requested gradient is computed.
>
> **G2 — Differentiation does not raise on a numerical condition.** Where a
> derivative does not exist, the gradient is **NaN**. This does not prohibit
> exceptions for unsupported backend operations or other non-numerical
> failures.
>
> **G3 — No host synchronisation** may be introduced to detect any condition in
> this section.
>
> **G4 — Only floating dtypes are differentiable.** Integer tensors cannot
> require gradients, so [12.4](#124-integer-exponentiation) never interacts
> with differentiation.
>
> **G5 — Each gradient carries its own operand's declared dtype**, not the
> upstream gradient's, and is reduced to that operand's shape.
>
> **G6 — Gradients execute on the selected backend** at every tensor size, or
> report that they cannot. No workload threshold, no silent fallback to another
> backend, and no decline that reads operand values.

#### 12.7.2 The region table

Each derivative is classified. The three classes are distinct and a convention
approved for one region is **not** extended to any other.

- **exists** — the partial derivative exists mathematically at that point and
  the stated value is it.
- **NaN** — no derivative exists; NaN records that fact.
- **convention** — no finite derivative exists; an approved representation of a
  one-sided infinite limiting slope.

| region | `f` | ∂f/∂x | class | ∂f/∂y | class |
| --- | --- | --- | --- | --- | --- |
| \(x > 0\) | \(x^{y}\) | \(y\,x^{y-1}\) | exists | \(x^{y}\ln x\) | exists |
| \(x < 0\), \(y\) integral | \(x^{y}\) | \(y\,x^{y-1}\) | exists | `NaN` | NaN |
| \(x < 0\), \(y\) non-integral | `NaN` | `NaN` | NaN | `NaN` | NaN |
| \(x = 0\), \(y > 1\) | `0` | `0` | exists | `0` | exists |
| \(x = 0\), \(y = 1\) | `0` | `1` | exists | `0` | exists |
| \(x = 0\), \(0 < y < 1\) | `0` | `+inf` | **convention** | `0` | exists |
| \(x = 0\), \(y = 0\) | `1` | `0` | exists | `NaN` | NaN |
| \(x = 0\), \(y < 0\) | `±inf` | `NaN` | NaN | `NaN` | NaN |
| \(x\) NaN or \(\pm\infty\) | IEEE | IEEE propagation | — | IEEE propagation | — |

Justifications for the rows that the formulas alone do not give:

- **\(x < 0\), \(y\) integral.** \(\partial f/\partial x = y\,x^{y-1}\) exists
  because \(y - 1\) is integral, so the power is real. This is the derivative
  **with respect to the base, holding the exponent fixed**. It is not a claim
  that \(f\) is differentiable as a function of two real variables at that
  point: \(\partial f/\partial y\) does not exist, because \(\ln x\) is
  undefined for \(x < 0\).
- **\(x = 0\), \(y > 0\): \(\partial f/\partial y = 0\).** \(f(0, y) = 0\) for
  every \(y > 0\), so \(f\) is constant in \(y\) there and the derivative is
  exactly zero. The formula's \(0 \cdot (-\infty)\) is a degenerate encoding of
  a derivative that genuinely exists.
- **\(x = 0\), \(y = 0\): \(\partial f/\partial x = 0\).** \(f(x, 0) = 1\) for
  every \(x\), so \(f\) is constant in \(x\) there. The formula's
  \(0 \cdot (+\infty)\) is again degenerate. \(\partial f/\partial y\) is NaN:
  \(f(0, y)\) is \(1\) at \(y = 0\) and \(0\) for \(y > 0\), so it is
  discontinuous and no derivative exists.
- **\(x = 0\), \(0 < y < 1\): \(\partial f/\partial x = +\infty\).** The
  right-hand difference quotient diverges to \(+\infty\). **No finite classical
  derivative exists at this point.** `+inf` is an approved numerical convention
  representing that one-sided infinite slope, and nothing more.
- **\(x = 0\), \(y < 0\): both NaN.** The forward result is an infinity, which
  does **not** establish that any derivative exists at the evaluation point.
  Neither partial derivative exists, and both are NaN. The signed-zero
  behaviour of the forward result is specified in
  [12.3.2](#1232-the-divide-by-zero-class) and is a separate matter.

#### 12.7.3 Worked example

```text
(-2.0) ** 3.0, upstream gradient 1.0

  forward          = -8.0
  base gradient    = 3 * (-2)^2 = 12.0      exists
  exponent gradient = NaN                    ln(-2) is undefined
```

Both are returned. The valid base gradient is **not** discarded because the
exponent gradient does not exist.

#### 12.7.4 Gradient accuracy

The bounds in [12.6.2](#1262-the-accuracy-bounds) apply to `**` itself. **They
are not imposed on a complete gradient expression**, which composes a power, a
logarithm and two multiplications, so error necessarily accumulates. Each
constituent operation meets whatever accuracy requirement its own
specification states; `pow`'s bounds are not transferred to `log` or to any
other operation by implication.

Ordinary differentiable cases are validated against a high-precision reference.
Finite-difference gradient checking is supplementary evidence only: it is
itself inaccurate near singularities and at non-differentiable points, and must
not be used to validate the NaN and convention rows of
[12.7.2](#1272-the-region-table).

*Historical behaviour (before D7, verified at the time):* a negative base
with a **tensor** exponent raised `ValueError` and discarded both gradients,
including the valid base gradient, while the same expression with a **scalar**
exponent returned `12.0` correctly. Zero-base cases raised. The checks read
both operands to the host, measured at 190 ms for a one-million-element
backward pass on CUDA, and the exponent-gradient kernel declined through
`bool(numpy.any(bases <= 0.0))`, a further synchronisation.

**Superseded.** `(-2.0) ** 3.0` now returns `12.0` for the base gradient and
`NaN` for the exponent gradient, together or separately, on all three
backends; every region of [12.7.2](#1272-the-region-table) is reachable; no
gradient reads an operand back to the host to classify it; and both power
gradients execute on the selected backend or raise.

### 12.8 Conformance requirements

In addition to [section 9](#9-conformance-testing-requirements):

1. **The special-value table** ([12.3.3](#1233-the-complete-special-value-table))
   — every row, both floating dtypes, every backend, compared exactly, with
   signed zeros distinguished and NaN compared by classification.
2. **Accuracy** ([12.6.2](#1262-the-accuracy-bounds)) — per element, against a
   validated reference built to [12.6.5](#1265-measurement), covering ordinary
   values, values near 1 with extreme exponents, subnormal results, and the
   overflow and underflow boundaries, at both precisions.
3. **Integer exponentiation** — wraparound at the promoted width for all five
   integer dtypes, and `ValueError` for every negative exponent including
   bases `1` and `-1`.
4. **Result dtypes** — all 49 cells of [12.5.1](#1251-tensor-base-with-tensor-exponent),
   both scalar directions, and every `cast` cell raising. Tests must assert
   that the result dtype does not change when only element **values** change.
5. **Differentiation** — every region of [12.7.2](#1272-the-region-table),
   asserting that a valid base gradient survives an undefined exponent
   gradient, and that gradient dtypes follow G5.
6. **Execution** — every rule above holds at every tensor size on every
   explicitly selected backend, and fused execution produces what unfused
   execution produces.

A conformance test must derive its expectations from this document, never from
the behaviour of a backend.

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
