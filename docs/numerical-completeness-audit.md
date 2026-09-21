# Numerical completeness audit

## 1. Executive summary and scope

### What was audited

The public numerical API of MS-Tensors at commit
`9b90ec49deeeb7a099e784301fbf0218a1ddb0e6`, against the repository itself
rather than against the record of completed milestones. The audit covered the
public exports, the operation implementations, the three backend kernel trees,
the dispatch layer, the fusion planner and generated CUDA source, the
differentiation rules, and the test suite.

Every claim below marked **demonstrated** is supported by a command and its
output, reproduced in this document. Claims marked **hypothesis** were not
established and are listed as work, not as findings.

### The headline

**The package has five numerical specifications, and together they govern
nine of its sixty public numerical operations.**

```
public operation functions:                60
governed by a numerical specification:      9   (+, -, *, /, **, sign, abs, sqrt, relu)
ungoverned:                                51
```

`sign`, `abs`, `sqrt` and `relu` are governed by their own documents
([sign-semantics.md](sign-semantics.md),
[abs-semantics.md](abs-semantics.md),
[sqrt-semantics.md](sqrt-semantics.md),
[relu-semantics.md](relu-semantics.md)). The audit
was originally written at five governed and fifty-five ungoverned; only the
counts have been restated, not the findings that rest on them.

**Forward and differentiation coverage are counted separately**, because
they are separate contracts and a count that merges them would hide which
one is missing. These four now have both; the other five governed
operations have forward coverage only. Migrating a VJP adds no public
operation function, so the count of sixty is unchanged by it.

```
public operation functions:                60
forward governed:                           9   (+, -, *, /, **, sign, abs, sqrt, relu)
first-order VJP governed:                   4   (sign, abs, sqrt, relu)
```

For those four, "VJP governed" means the first-order VJP, the graph-built
VJP and the higher-order regions each document names — not that autodiff
through them is specified without limit. `sqrt` records one measured limit
of its own in [section 8.1](sqrt-semantics.md#81-a-limit-on-third-and-higher-order-measured-here).

Within its scope the specification is in good order: the D1–D7 and S3
milestones are implemented, tested against validated high-precision
references, and the full suite passes (1,782 tests, 0 failures, 0 expected
failures, on `python`, `numpy` and `cuda`). Nothing in this audit reopens
them.

Outside that scope there is no numerical contract at all. Fifty-one public
operations have no stated accuracy bound, no exceptional-value table, no
signed-zero requirement, no subnormal requirement, and no rule about whether
fused and eager execution must agree. Their tests establish that the backends
agree **with each other**, which is a different and much weaker claim.

That gap is not theoretical. Auditing it surfaced one demonstrated numerical
defect affecting twelve operations on CUDA, and one demonstrated conflict
between two promotion authorities that silently corrupts integer values.

### Two demonstrated defects

**D-1 — a numerical implementation defect. Eager CUDA binary32 elementwise
kernels flush subnormals to zero.** `ts.abs` of the smallest binary32
subnormal returns `0.0` on CUDA and the correct value on Python and NumPy.
`abs` and `sign` are **exact** operations (class E): on ordinary finite inputs
their results are determined, not approximated, so `ts.sign` of a positive
subnormal returning `0.0` instead of `1.0` is an **incorrect result**, not an
accuracy shortfall, and no tolerance would excuse it. Twelve operations are
affected. The *fused* path is correct, so the same expression gives different
answers depending on whether it was fused.

**D-2 — a semantic-authority conflict. Two promotion authorities disagree on
four of forty-nine dtype cells.** `int64 + float32` raises
`DtypePromotionError`, because no floating dtype holds every `int64` value.
`ts.maximum` on the same operands silently promotes to `float64` and loses the
value. This is equally demonstrated, and equally a defect; it differs from D-1
in kind, because no single component is wrong — two components disagree, and
which one should yield is a policy question that has not been decided.

### What "numerically complete" would require

Not more tests. A decision, per operation, on which of the five numerical
classes in §2 applies, followed by a specification, a reference, and tests
derived from the specification rather than from another backend. §9 gives a
dependency-ordered plan; §10 gives the completion criteria.

---

## 2. The numerical classes used in this audit

Each operation is classified into one of five kinds of numerical behaviour.
The class determines what a conformance test may assert.

| Class | Meaning | What a test may assert |
| --- | --- | --- |
| **E** Exact | integer arithmetic, or a floating result that is exact | bitwise equality, across backends |
| **R** Correctly rounded | the result is uniquely determined by IEEE 754 | bitwise equality, across backends |
| **A** Approximate | a stated error bound against a validated reference | per-element ULP bound; **not** cross-backend equality |
| **C** Classified | NaN, ±inf, ±0, domain and overflow behaviour | exact classification, across backends |
| **N** Nondeterministic | intentionally unreproducible, or reproducible only under stated conditions | the stated reproducibility conditions only |

### Kinds of finding

The report distinguishes five kinds of finding, and uses these terms
consistently:

| Term | Meaning | Example |
| --- | --- | --- |
| **Implementation defect** | a component computes a result its own contract forbids | D-1 |
| **Semantic-authority conflict** | two components define conflicting behaviour and no policy says which governs | D-2 |
| **Missing specification** | no contract exists, so no result can be called conforming or not | S-1 to S-6 |
| **Missing test evidence** | a contract exists but nothing establishes that it holds | T-1 to T-5 |
| **Hypothesis** | stated but not established by this audit; listed as work | the FTZ cause of D-1 |

A missing specification is not a defect, and an operation is never called
nonconforming merely for being unspecified. Conversely, a demonstrated defect
is not downgraded because the specification that would name it is absent:
D-1's `sign` result is wrong on the operation's own terms.

Two rules follow from the principles in the task, and this audit applies them
throughout:

- **A backend is never the authority.** A test that compares NumPy against
  Python establishes that they agree, not that either is right.
- **Internal consistency is necessary but insufficient.** A uniformly
  inaccurate implementation is consistently wrong.

---

## 3. Operation inventory

Discovered from `tensors.__all__` and the submodules, not assumed.

### 3.1 Governed by `docs/arithmetic-semantics.md`

| Operation | Location | Class |
| --- | --- | --- |
| `+` `add` | `operations/arithmetic/add.py` | R, C |
| `-` `subtract` | `operations/arithmetic/subtract.py` | R, C |
| `*` `multiply` | `operations/arithmetic/multiply.py` | R, C |
| `/` `divide` | `operations/arithmetic/divide.py` | R, C |
| `**` `pow` | `operations/arithmetic/power.py` | E (integer), A (2/4 ULP), C |

### 3.2 Ungoverned — 51 operations

| Family | Operations | Location |
| --- | --- | --- |
| Elementary | `exp` `log` | `operations/elementary/` |
| Trigonometric | `sin` `cos` `tan` `arcsin` `arccos` `arctan` | `operations/trigonometric/` |
| Hyperbolic | `sinh` `cosh` `tanh` `arcsinh` `arccosh` `arctanh` | `operations/hyperbolic/` |
| Activations | `sigmoid` `softplus` | `operations/activations/` |
| Comparison | `equal` `not_equal` `less` `less_equal` `greater` `greater_equal` | `operations/comparison/` |
| Selection | `where` `clip` `maximum` `minimum` | `operations/selection/` |
| Reductions | `sum` `mean` `prod` `max` `min` `std` `variance` `norm` `logsumexp` `argmax` `argmin` | `operations/reductions/` |
| Normalization | `softmax` `log_softmax` | `operations/normalization/` |
| Losses | `binary_cross_entropy` `cross_entropy` | `operations/losses/` |
| Linear algebra | `matmul` `dot` `outer` `transpose` `norm` | `operations/linalg/`, `linalg/` |
| Convolution | `conv1d` `conv2d` `conv3d` | `operations/convolution/` |
| Manipulation | `concat` `stack` `reshape` | `operations/manipulation/` |

The elementary family does not share one numerical class, and the distinction
matters for what a conformance test may assert:

| Operation | Class | What that requires |
| --- | --- | --- |
| `abs` | **E**, **C** | exact on ordinary finite inputs; no tolerance applies |
| `sign` | **E**, **C** | exact on ordinary finite inputs; no tolerance applies |
| `sqrt` | **R**, **C** | correctly rounded — IEEE 754 determines it uniquely |
| `exp` | **A**, **C** | a stated error bound against a validated reference |
| `log` | **A**, **C** | a stated error bound against a validated reference |

The **C** component of each is a separate question. The exceptional-value,
NaN and signed-zero rules for these operations are **not decided here**; they
belong to the specification work in §9, Phase 4.1. What is settled is that
`abs` and `sign` are exact on ordinary finite inputs, so an approximation
tolerance is never the right instrument for them.

### 3.3 Adjacent numerical surface, also ungoverned

| Surface | Entries | Numerical question |
| --- | --- | --- |
| Creation | `arange` `linspace` `eye` `full` `ones` `zeros` | `linspace` endpoint exactness; `arange` accumulation |
| Casting | `Tensor.astype`, `casting.cast_values` | rounding mode; out-of-range integer conversion |
| Random | `random.normal` `uniform` `randint` `seed` | class **N**: distribution and reproducibility conditions |
| Initialization | 9 initializers in `ts.init` | derived from `random`; fan computation exactness |
| Optimizers | `SGD` `Adam` `RMSprop` | accumulation order, epsilon placement, state dtype |
| Autodiff | `backward` `grad` `jacobian` `hessian` `gradcheck` | per-operation derivative rules; only `**` has one |

### 3.4 Not public API

`ts.ops` and `ts.math` re-export operation *classes* (`Add`, `Sin`, …). These
are the internal `Operation` objects, not a second numerical surface.
`tensors/utils/` holds shared primitives (broadcasting, summation, stability
helpers) that carry numerical behaviour but are not independently public.

---

## 4. Conformance matrix

Condensed to the fields that differ. **Spec** is the governing document;
**Acc** is a stated error bound; **Exc** is an exceptional-value contract;
**Sub** is a subnormal requirement; **Bk** is backend-parametrised operation
tests; **Fus** is a fused-vs-eager requirement.

| Operation family | Spec | Acc | Exc | Sub | Bk | Fus | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `+ - * /` | §§1–11 | R | yes | §5.4 | yes | yes | **Verified** |
| `**` | §12 | 2/4 ULP | §12.3.3 | §5.4 | yes | yes | **Verified** |
| `**` gradients | §12.7 | detection only | §12.7.2 | yes | yes | yes | **Verified** |
| Elementary `sign` (forward) | [sign-semantics.md](sign-semantics.md) | n/a (class E) | n/a | §1.5 | yes | yes | **Governed.** Forward only; `sign`'s differentiation remains ungoverned |
| Elementary `abs` (forward) | [abs-semantics.md](abs-semantics.md) | n/a (class E) | §1.7 | §1.6 | yes | yes | **Governed.** Forward only; `abs`'s differentiation remains ungoverned |
| Elementary `sqrt` (forward) | [sqrt-semantics.md](sqrt-semantics.md) | correctly rounded (§1.2) | §1.4 **no longer traps** | §1.7 | yes | yes | **Governed.** Forward only; `sqrt`'s differentiation remains ungoverned and still traps at zero |
| Elementary `exp` `log` | — | — | traps | — | no | no | **Spec missing** |
| Trigonometric | — | — | traps | — | no | no | **Spec missing** |
| Hyperbolic | — | — | traps | — | no | no | **Spec missing** |
| Activations `relu` (forward) | [relu-semantics.md](relu-semantics.md) | n/a (class E) | n/a (§1.6) | §1.4 | yes | yes | **Governed.** Forward only; `relu`'s differentiation remains ungoverned |
| Activations `sigmoid` `softplus` | — | — | — | — | no | no | **Spec missing** |
| Comparison | — | n/a | — | n/a | no | n/a | **Spec missing** |
| Selection | — | n/a | — | n/a | no | n/a | **Spec missing**, see D-2 |
| Reductions | — | — | — | — | no | n/a | **Spec missing** |
| Normalization | — | — | — | — | yes | n/a | **Spec missing** |
| Losses | — | — | — | — | no | n/a | **Spec missing** |
| Linear algebra | — | — | — | — | no | n/a | **Spec missing** |
| Convolution | — | — | — | — | yes | n/a | **Spec missing** |
| Manipulation | — | n/a | n/a | n/a | no | n/a | Dtype only, see D-2 |
| Creation | `initialization.md` (partial) | — | — | — | no | n/a | **Spec missing** |
| Random | — | — | n/a | n/a | no | n/a | **Spec missing** (class N) |
| Optimizers | — | — | — | — | no | n/a | **Spec missing** |

No ungoverned operation is classified *nonconforming* on the strength of
having no specification. D-1 is classified nonconforming for three
independent reasons, none of which depends on the missing specification: it
violates §5.4, which the arithmetic specification states for the package's
floating behaviour generally; two execution paths of the same package
disagree, which `docs/autodiff.md` forbids; and `abs` and `sign` are exact
operations whose results on ordinary finite inputs are determined, so CUDA
returns values that are simply wrong rather than insufficiently accurate.

---

## 5. Confirmed defects

### D-1 — Eager CUDA binary32 elementwise kernels flush subnormals

**Status when audited: demonstrated.** Twelve operations. **Re-measured on
2026-09-21: the reproducer no longer reproduces** — see
[the re-measurement](#d-1-re-measured-2026-09-21) below, which supersedes the
counts in this section. The finding is kept as written because the cause it
identifies is still live one layer down.

```python
import tensors as ts
SMALLEST = 1.401298464324817e-45          # smallest positive binary32 subnormal

for backend in ts.available_backends():
    with ts.use_backend(backend):
        t = ts.Tensor([SMALLEST], dtype=ts.float32)
        print(backend, ts.abs(t).tolist()[0], ts.sign(t).tolist()[0])
```

```
python  1.401298464324817e-45 1.0
numpy   1.401298464324817e-45 1.0
cuda    0.0                   0.0
```

`abs(x)` of a positive subnormal is `x`; `sign(x)` is `1.0`. CUDA returns zero
for both.

Both are **class E** operations: on ordinary finite inputs their results are
exact, so these are **incorrect results**, not accuracy shortfalls. `sign`
additionally returns the wrong classification — the sign of a positive value
is reported as neither positive nor negative. No error bound, however
generous, would make either of these conforming.

Affected, measured at the smallest binary32 subnormal: `sqrt`, `abs`, `sign`,
`sin`, `tan`, `arcsin`, `arctan`, `arcsinh`, `arctanh`, `sinh`, `tanh`,
`relu`. `float64` is unaffected on every backend.

**The fused path is already correct**, so the same expression disagrees with
itself depending on whether the planner fused it:

| operation | eager CUDA | fused CUDA |
| --- | --- | --- |
| `sqrt` | `0.0` | `3.743392066509216e-23` |
| `sign` | `0.0` | `1.0` |
| `abs`, `sin`, `tan`, `arcsin`, `arctan`, `arcsinh`, `arctanh`, `sinh`, `tanh`, `relu` | `0.0` | `1.401298464324817e-45` |

**Cause (hypothesis, not yet established):** this is the same flush-to-zero
behaviour corrected for `+ - * /` under D2 and for `**` under the CUDA
gradual-underflow work — NVRTC applies FTZ to binary32 instructions and
ignores `--ftz=false`. The eager elementwise kernels were never given the
inline-PTX treatment that `ieee32.py` applies to arithmetic, and the fused
generator was corrected separately. This should be confirmed against each
kernel before implementation.

**Why it matters beyond the values:** §5.4 requires gradual underflow, and
`docs/autodiff.md` requires fused execution to produce what unfused execution
produces. Both are violated.

#### D-1 re-measured (2026-09-21)

The reproducer above was rerun unchanged on `feat/backend-residency-dispatch`
before the forward `sign` migration. **All twelve operations now agree with
the Python backend at the smallest binary32 subnormal**, `sign` and `abs`
included:

```
python  1.401298464324817e-45 1.0
numpy   1.401298464324817e-45 1.0
cuda    1.401298464324817e-45 1.0
```

`sqrt`, `abs`, `sign`, `sin`, `tan`, `arcsin`, `arctan`, `arcsinh`,
`arctanh`, `sinh`, `tanh` and `relu` were each measured; none flushes. The
eager/fused split the finding describes is therefore also closed for these
twelve. What changed is not the audit's cause but the kernels' route to it:
the eager elementwise kernels reach their operands through the CUDA
`conversion` boundary, whose `_working_values` widens binary32 through the
PTX conversion in `ieee32.py` rather than through `astype`. That widening
landed after this audit was written.

**The device behaviour the finding identified is unchanged.** Measured
directly, `cupy.sign` on a native binary32 array still returns `0.0` for
`±1.401298464324817e-45`, and so does every other binary32 ufunc under FTZ.
The defect is avoided by the conversion boundary, not eliminated at the
provider. Any kernel that classifies or computes *directly* in binary32 —
rather than widening first — reintroduces it. The forward `sign` migration
hit exactly this: lowering to a native binary32 array moved the operand past
`_working_values`, so
`tensors/backend/cuda/kernels/elementwise/sign.py` widens explicitly through
`ieee32.widen` and [sign-semantics.md](sign-semantics.md) §1.5 pins the
requirement with a test.

This re-measurement covers the reproducer only. It does not revisit the
accuracy findings (S-1 to S-3) or the coverage findings (T-1, T-2), which
were not rerun.

**`abs` re-measured again during its own migration (2026-09-21).** Before the
migration, `ts.abs` of the smallest binary32 subnormal returned
`1.401298464324817e-45` on all three backends, and eager agreed with fused;
after it, the same. Measuring the provider directly showed the finding's
cause intact in *both* directions for `abs`, not just on the way in:

```
cupy.abs(native binary32 [±smallest])   -> [0.0, 0.0]
binary64 magnitude, CuPy-converted down -> [0.0, -0.0]
ieee32.widen / ieee32.narrow            -> [±1.401298464324817e-45]
```

`abs` returns the operand's magnitude, which can itself be subnormal, so
unlike `sign` it needs the PTX conversion on the way out as well. The
migrated kernel uses both, and [abs-semantics.md](abs-semantics.md) §1.6
pins the requirement with a test.

**`sqrt` re-measured during its own migration (2026-09-21).** The same cause
is present on the operand side, and measuring it against an independent
reference showed a second, separate shortfall the subnormal reproducer does
not reach:

```
cupy.sqrt(native binary32 [smallest subnormal]) -> 0.0
cupy.sqrt(native binary32), 400 random operands -> 2 not correctly rounded
ieee32.widen -> binary64 sqrt -> ieee32.narrow -> correctly rounded, 3666/3666
```

The 3,666 cases include operands chosen so that their true roots sit as close
as possible to a binary32 rounding boundary — the cases a root computed in a
wider format and rounded down would get wrong. None failed, so **no dedicated
binary32 PTX square-root instruction was needed**: binary64's 53 significand
bits exceed the `2p + 2 = 50` that square root requires for the second
rounding to agree with rounding the true root directly. This is the first
finding in this audit measured against a reference for *correct rounding*
rather than for subnormal survival, and the second row above is a defect the
D-1 reproducer would never have surfaced.

`cupy.sqrt` on binary64 was correctly rounded on all 2,002 operands measured.

**The four VJPs re-measured during their own migration (2026-09-21).** Two
findings, neither of which the forward reproducers could reach.

*A cross-backend disagreement in the VJPs themselves.* The Python kernels
routed and returned a literal zero where the array kernels materialised a
derivative and multiplied. At a finite nonzero primal the sign VJP gave:

```
g = -2.0   ->  python +0.0   numpy -0.0   cuda -0.0
g = ±inf   ->  python +0.0   numpy  nan   cuda  nan
g = nan    ->  python +0.0   numpy  nan   cuda  nan
```

`abs` and `relu` disagreed the same way on their inactive branches. All four
specifications settle it in favour of routing, and the three backends now
agree.

*Flush-to-zero reaches comparisons, not only arithmetic.* This is the part
D-1's reproducer could not show, because it only ever measured values. On
native binary32:

```
subnormal == 0.0   ->  True     (so the sign and sqrt VJPs would raise
                                 "undefined at zero" for a nonzero primal)
subnormal >  0.0   ->  False    (so the whole positive subnormal band would
                                 be routed to ReLU's inactive branch)
-g for subnormal g ->  ∓0.0     (so abs would lose a subnormal upstream on
                                 its negation branch, but not on the other)
```

Each is corrected by widening the operand that the *predicate* or the
negation reads. Selection itself needs no protection: a `where` preserves a
subnormal, which is why ReLU's VJP widens only its primal and leaves the
upstream native.

**`relu` re-measured during its own migration (2026-09-21).** The same cause
again, on both sides of the format boundary, and it reaches further than the
D-1 reproducer's single smallest-subnormal probe:

```
cupy.maximum(native binary32 [smallest subnormal], 0) -> 0.0
cupy.maximum(native binary32 [largest  subnormal], 0) -> 0.0
ieee32.widen -> maximum in binary64 -> ieee32.narrow -> both preserved
```

The **largest** binary32 subnormal flushes as readily as the smallest, so the
affected input range is the whole subnormal band rather than its lower edge.
`relu` returns its operand, so like `abs` the result can itself be subnormal
and the PTX conversion is needed in both directions; the migrated kernel uses
both, and [relu-semantics.md](relu-semantics.md) §1.4 pins it with a test.

Two further behaviours were measured while choosing the implementation, and
neither is a defect: `cupy.maximum` and `numpy.maximum` both propagate NaN
and both return canonical positive zero for `-0.0`. A `where(x > 0, x, 0)`
selection does neither for NaN — it sends NaN to the zero branch — which is
why the maximum form was chosen. Eager and fused CUDA `relu` agree, with the
fused kernel executing rather than declining.

### D-2 — Two promotion authorities disagree

**Status: demonstrated.** Four of forty-nine dtype cells.

`tensors/dtype.py` contains two promotion functions. `arithmetic_result_dtype`
implements the approved §6.2 table; the older `result_dtype` is still used by
comparison, selection, `where`, `clip`, `concat`, `stack`, `matmul`, `outer`,
the losses, convolution and the SGD update — fourteen modules.

```python
import tensors as ts
big = 2**62 + 1                            # exact in int64, not in float64
a = ts.Tensor([big], dtype=ts.int64)
b = ts.Tensor([1.5], dtype=ts.float32)
```

```
a + b             -> DtypePromotionError: no supported floating dtype represents every int64 value
ts.maximum(a, b)  -> float64  4.611686018427388e+18
ts.minimum(a, b)  -> float64  1.5
ts.where(m, a, b) -> float64  4.611686018427388e+18
ts.concat([a, b]) -> float64  4.611686018427388e+18
ts.stack([a, b])  -> float64  4.611686018427388e+18
```

The arithmetic contract refuses the combination precisely because `float64`
cannot represent every `int64` value. The legacy path performs the same
conversion silently. `4611686018427387905` becomes `4611686018427387904`.

The four disagreeing cells are `int64 ⊕ float32`, `int64 ⊕ float64` and their
transposes: legacy gives `float64`, the arithmetic contract raises.

This is a *semantic authority* conflict, not merely duplication. Two
components define conflicting behaviour for the same operand pair.

---

## 6. Missing or ambiguous specifications

### S-1 — No accuracy contract for any transcendental function

**Status: demonstrated (the absence); measurements are evidence of size, not
of a violation.**

Measured cross-backend spread against the Python backend, 4,096 deterministic
values, maximum ULP:

| | float64 | float32 |
| --- | --- | --- |
| `exp` `log` `sin` `cos` `tan` `arccos` `arctan` `sinh` `cosh` `tanh` `softmax` | 1 | 0 |
| `arcsin` `arcsinh` `arctanh` `arccosh` `sigmoid` `softplus` `norm` | 2 | 0 |
| `sqrt` `abs` `sign` `relu` `max` `min` `std` `variance` `log_softmax` `logsumexp` | 0 | 0 |

These differences are **permitted by legitimate algorithmic variation** and
are not defects. The finding is that nothing says so: there is no bound they
are permitted *within*, and no validated reference establishing that any of
the three is correct. A backend could drift to 50 ULP and no test would fail.

Three entries in the last row are not class A and are listed there only
because they were measured alongside the rest:

- `sqrt` is **correctly rounded** (class R) under IEEE 754, so cross-backend
  bitwise equality is *required* there rather than merely observed.
- `abs` and `sign` are **exact** (class E) on ordinary finite inputs. Neither
  needs an accuracy bound; both need an exactness test and a classification
  rule. Their 0 ULP measurement is therefore the required result, not a
  favourable observation — and it holds only for the operands measured here,
  which did not include subnormals. D-1 shows CUDA failing both.

### S-2 — No accumulation-order contract for reductions

**Status: demonstrated.**

| reduction | float64 max ULP vs Python | float32 |
| --- | --- | --- |
| `sum` | 4 | 0 |
| `mean` | 4 | 0 |
| `prod` | 10 | 0 |
| `norm` | 2 | 0 |

Different accumulation orders are a legitimate implementation choice, and
§6 of the task is explicit that bitwise agreement must not be demanded where
the contract permits it. But no contract permits it, because there is no
contract. What is needed is a decision per reduction: pairwise or sequential,
whether a compensated algorithm is required, and an error bound as a function
of element count — not a tolerance chosen to make today's numbers pass.

`max`, `min`, `std`, `variance`, `logsumexp` measure 0 ULP, which suggests
`std`/`variance`/`logsumexp` use a shared stabilised primitive. Confirming
that and specifying it is part of the same work.

### S-3 — Trapping policy is inconsistent and unstated

**Status: demonstrated.**

Domain violations in the elementary functions **raise**, identically on all
three backends:

```
log(0) log(-1) arcsin(2) arccosh(0) arctanh(1) sin(inf)  -> ValueError
```

**Corrected 2026-09-21.** `sqrt(-1)` was in that list when this finding was
written and is no longer: forward `sqrt` now returns `nan`, and
[sqrt-semantics.md](sqrt-semantics.md) §1.4 specifies it as a value rather
than a domain error. That settles the question for `sqrt` alone and **leaves
this finding open** for every function still listed above — the package-wide
trapping policy is still unstated, which is what S-3 is about.

Arithmetic and exponentiation went the other way: D2 made every exceptional
value a **result**, explicitly so that no host synchronisation is needed to
detect it and so that fused and eager agree.

Both policies are defensible; the package currently holds both without saying
which applies where. This must be decided **before** any elementary function
is specified, because the answer determines the whole shape of its
exceptional-value table and whether its CUDA kernel may read operands back.

Cross-backend exceptional-value agreement was measured across 22 cases and is
currently **0 disagreements** — a good baseline to specify against.

### S-4 — Signed zero is unspecified outside arithmetic

**Corrected 2026-09-21; narrowed, not closed.** When this finding was written
nothing outside arithmetic said what a signed zero should produce. Three
operations have since been specified and are no longer examples of it:

- `sign(-0.0)` and `abs(-0.0)` return canonical `+0.0`, required by
  [sign-semantics.md](sign-semantics.md) §1.3 and
  [abs-semantics.md](abs-semantics.md) §1.3.
- `sqrt(-0.0)` returns `-0.0`, required by
  [sqrt-semantics.md](sqrt-semantics.md) §1.3, which keeps the sign
  deliberately because IEEE 754 defines it that way for a root.

That those three do not agree with each other is the point: each was decided
on its own operation's terms. **The finding stands for everything else** —
`max(-0.0, 0.0)`, `sum([-0.0, -0.0])`, the reductions and the comparison and
selection families still have no stated rule.

### S-5 — No specification for the adjacent surface

Creation (`linspace` endpoint exactness, `arange` accumulation), casting
(rounding mode, out-of-range integer conversion), random sampling
(distribution contract and the reproducibility conditions of class N),
initializers, and optimizer update arithmetic all lack numerical contracts.
`docs/initialization.md` covers initializer *selection*, not numerical
behaviour.

### S-6 — Derivative rules are specified only for `**`

§12.7 gives `**` a region table, six rules, and a classification scheme.
Every other differentiable operation has derivative code but no stated
contract: no region table at non-differentiable points (`abs` at 0, `relu` at
0, `sign` everywhere, `max`/`min` at ties, `clip` at its bounds), no accuracy
statement, and no independence rule.

---

## 7. Missing test coverage

### T-1 — Operation tests do not select a backend

**Status: demonstrated.** The default backend at import is `python`.

| family | test files parametrised over backends |
| --- | --- |
| `trigonometric` | 0 / 3 |
| `hyperbolic` | 0 / 3 |
| `comparison` | 0 / 2 |
| `elementary` | 0 / 4 |
| `reductions` | 0 / 7 |
| `selection` | 0 / 3 |
| `losses` | 0 / 2 |
| `linalg` | 0 / 3 |
| `activations` | 0 / 2 |
| `manipulation` | 0 / 5 |
| `normalization` | 2 / 2 |
| `convolution` | 2 / 2 |
| `arithmetic` | 19 / 19 |

**31 of 36 operation test files exercise the Python backend only.** The NumPy
and CUDA kernels for those families are reached only through
`tests/backend/`, which is parity-based — see T-2. This is why D-1 survived:
no test in `tests/operations/elementary/` ever ran on CUDA.

### T-2 — Backend tests use one backend as the oracle

**Status: demonstrated.** `tests/backend/_support.py` defines
`NumPyParityTestCase`:

```python
def assertOperationParity(self, function):
    expected = self._evaluate("python", function)
    actual = self._evaluate("numpy", function)
    ...
    self.assertEqual(actual.tolist(), expected.tolist())
```

Used by ten modules. This is exactly the pattern §7 of the task asks to
identify: it compares two implementations without establishing that either is
correct, and it demands **bitwise** equality while doing so — a requirement no
document states and which the class-A operations should not be held to.

It is not worthless: it would catch a gross NumPy regression. It cannot
establish conformance, and it silently encodes the Python backend as the
specification.

### T-3 — No independent reference outside arithmetic

The validated high-precision reference infrastructure built for D5 and D7
(`tests/operations/arithmetic/_reference.py`, `_gradient_reference.py`,
`_accuracy.py`) is excellent and is used by nothing outside `**`. Every other
operation's expectations are literals, backend outputs, or `math` module
results.

### T-4 — No fused-versus-eager coverage for 20 of 26 fusible operations

The fusion planner fuses 26 operations. The arithmetic contract governs six.
The remaining twenty have no test requiring fused and eager to agree — which
is how the D-1 split (eager wrong, fused right) went unnoticed.

### T-5 — Specific gaps worth naming

- No subnormal test for any operation outside `+ - * / **`.
- No test that `sqrt` is correctly rounded (class R), and none that `abs` or
  `sign` are exact (class E).
- No reduction test at sizes where accumulation order matters (the audit used
  4,096 elements; the suite's reduction tests are much smaller).
- No reproducibility test for `random` under fixed seed across backends.
- No gradient-accuracy test outside `**`.

---

## 8. Documentation inconsistencies

### X-1 — `docs/arithmetic-semantics.md` §1 says `**` is not implemented

**Status: demonstrated.** The opening status block reads:

> **Exponentiation (`**`) is specified in [section 12](#12-exponentiation)
> but is not implemented.** That section is the approved target contract; the
> package still behaves as its *Current behaviour* passages record.

D1–D7 and S3 are implemented, reviewed and approved. Every *Current
behaviour* passage in §12 is now historical. §12.7.4's "Current behaviour
(verified)" paragraph describing `ValueError` on a negative base with a tensor
exponent is contradicted by the code and by the D7 tests.

### X-2 — The same block's scope statement is stale

It states that every operation other than `+ - * /` "still uses the older
promotion in `tensors/dtype.py` `result_dtype`". That is true for fourteen
modules but no longer true for `**`, and the sentence gives no hint that the
divergence is a live conflict (D-2).

### X-3 — Two gaps recorded in §1 need re-checking

§1 records the observability API as absent — still true — and the VJP
execution-location requirement as unmet "for the kernels named in §10.3".
Power's gradients now have strict dispatchers, so §10.3's list needs
re-auditing against the code. Not investigated here.

`docs/backends.md` was reconciled in the previous milestone and is current.

---

## 9. Unsupported functionality and scope exclusions

Distinguished from defects, per §4 of the task.

| Item | Status |
| --- | --- |
| Floor division, integer division operator | **Not supported.** Explicitly out of scope in §1 of the arithmetic spec. Not a defect. |
| Complex dtypes | **Not supported.** Seven public dtypes, all real. |
| `float16`, `bfloat16` | **Not supported.** |
| CuPy object-dtype integer intermediates | **Not supported** by CuPy; some non-arithmetic kernels use the Python path for exact integer work. Legitimate capability gap, documented in `backends.md`. |
| Execution-observability API | **Absent by record.** Required by `backends.md`; outstanding independently. Excluded from this audit. |
| Multi-device / multi-GPU | **Not supported.** No numerical question arises. |

---

## 10. Implementation plan, dependency-ordered

Ordered by what blocks what, not by effort.

### Phase 0 — Record correctness (no code)

**P0.1** Correct `docs/arithmetic-semantics.md` §1 and the stale *Current
behaviour* passages in §12 (X-1, X-2). Re-audit §10.3 against the code (X-3).

*Blocks nothing technically, but the specification is the authority and it
currently misstates what is implemented.*

### Phase 1 — Decisions that block all specification work

**P1.1 Trapping policy (S-3).** Decide whether elementary and transcendental
functions trap on domain violations or return classified values. Every
subsequent exceptional-value table depends on this, as does whether a CUDA
kernel may read operands back to decide.

**P1.2 Promotion authority (D-2).** Decide whether the §6.2 table governs the
whole package. Until this is settled, no comparison, selection, reduction or
manipulation operation can have its dtype behaviour specified coherently,
because two authorities would answer differently. The demonstrated silent
precision loss makes this urgent as well as blocking.

**P1.3 Signed-zero and subnormal policy (S-4).** State once, package-wide,
what §5.4 requires of every floating operation, so it need not be restated per
operation.

### Phase 2 — The CUDA implementation defect

**P2.1 Fix D-1.** Confirm the FTZ hypothesis per kernel, then apply the
established inline-PTX remedy to the twelve eager CUDA binary32 elementwise
kernels. Depends on P1.3 for the requirement, not for the technique — the
technique already exists in `ieee32.py`.

*D-1 is the audit's demonstrated **implementation** defect: one backend
computes results its own package says are wrong. It produces incorrect
class-E results and a wrong `sign` classification, not merely inaccurate
values, so it is remediable without any new policy decision — the correct
answers are already fixed by the operations' exactness.*

*The audit's other demonstrated defect, D-2, is deliberately **not** in this
phase.* It is a semantic-authority conflict rather than an implementation
error: no single component computes a wrong answer on its own terms, and
until P1.2 decides which authority governs, there is no correct behaviour to
implement. **The fourteen modules using the legacy promotion rule must not be
changed before that policy is agreed**, and D-2's remediation is a separate
milestone from P2.1 — the two share no code and no decision.

### Phase 3 — Reference infrastructure

**P3.1** Generalise `tests/operations/arithmetic/_reference.py` into a
package-wide high-precision reference covering `exp`, `log`, the
trigonometric and hyperbolic families, and their inverses, with the same
enclosure discipline: refine precision until the rounding resolves, and refuse
to answer otherwise.

**P3.2** Generalise `_accuracy.py`'s ULP metric and classification rules for
reuse. They are already operation-agnostic.

*Blocks every accuracy specification, because a bound cannot be stated without
something to measure against.*

### Phase 4 — Specify, family by family

In this order, because each depends on the primitives of the previous:

**P4.1** Elementary (`abs`, `sign`, `sqrt`, `exp`, `log`) — **not one class**:
`abs` and `sign` as class E, `sqrt` as class R, `exp` and `log` as class A,
each with its own class-C rules. Smallest family, and `exp`/`log` underpin the
rest. The class-E and class-R members need exactness and correct-rounding
tests rather than a tolerance; only `exp` and `log` need the reference
infrastructure of Phase 3, so the exact members can be specified first.
**P4.2** Trigonometric and hyperbolic, and their inverses.
**P4.3** Comparison and selection — mostly class C and dtype rules; depends
on P1.2.
**P4.4** Reductions (S-2) — accumulation contract and element-count-dependent
bounds; depends on P4.1 for `logsumexp`, `std`, `norm`.
**P4.5** Normalization and losses — compositions of P4.1 and P4.4.
**P4.6** Linear algebra and convolution — accumulation contracts again, larger
error budgets.
**P4.7** Creation, casting, random (class N), initializers, optimizers.

### Phase 5 — Differentiation

**P5.1** Extend §12.7's region-table discipline to every differentiable
operation (S-6), starting with the non-differentiable points: `abs` and `relu`
at zero, `sign`, `max`/`min` at ties, `clip` at its bounds.

### Phase 6 — Testing

Runs alongside Phases 4–5, not after.

**P6.1** Replace `NumPyParityTestCase` with specification-derived tests as
each family is specified (T-2). Retire it only when its modules are covered.
**P6.2** Parametrise the operation test suites over backends (T-1).
**P6.3** Add fused-versus-eager coverage for the twenty ungoverned fusible
operations (T-4).
**P6.4** Close the named gaps in T-5.

---

## 11. Completion criteria

Numerical completeness may be declared when **all** of the following hold. No
single one is sufficient, and a passing test suite is not among them.

1. **Every public numerical operation is classified** into one of E, R, A, C
   or N, and the classification is recorded in a specification document.
2. **Every class-A operation has a stated error bound** and a validated
   independent reference that the bound is measured against — not another
   backend, and not a fixed-precision calculation without an enclosure. An
   error bound is stated for class-A operations *only*; attaching one to a
   class-E or class-R operation would weaken a requirement, not record it.
3. **Every class-E operation is tested for exactness and every class-R
   operation for correct rounding**, bitwise, on every backend.
4. **Every floating operation has an exceptional-value table** covering NaN,
   ±inf, ±0, domain violations and overflow, compared exactly with NaN by
   classification.
5. **The subnormal requirement of §5.4 is stated package-wide and tested** for
   every floating operation on every backend.
6. **One promotion authority** governs the package, and no two components
   answer the same dtype question differently.
7. **One trapping policy** governs the package, stated explicitly.
8. **Every operation's tests select each backend explicitly** and assert that
   the selected backend's kernel executed — not that the result was plausible.
9. **Fused execution is required to produce what unfused execution produces**,
   for every fusible operation, and is tested at subnormal and exceptional
   inputs.
10. **Every differentiable operation has a derivative contract** covering its
    non-differentiable points, with the three-class discipline of §12.7.2.
11. **No test uses one backend as the oracle for another.**
12. **No global floating-point tolerance exists anywhere in the suite.**

### Current position against these criteria

| Criterion | `+ - * /` | `**` | Other 51 |
| --- | --- | --- | --- |
| 1 Classified | met | met | not met |
| 2 Error bound + reference | n/a (class R) | met | not met |
| 3 Exactness / correct rounding tested | met | n/a | not met (`abs`, `sign`, `sqrt`) |
| 4 Exceptional-value table | met | met | not met |
| 5 Subnormals | met | met | **violated** (D-1) |
| 6 One promotion authority | — | — | **violated** (D-2) |
| 7 One trapping policy | — | — | **violated** (S-3) |
| 8 Backend-explicit tests | met | met | not met (T-1) |
| 9 Fusion contract | met | met | not met (T-4) |
| 10 Derivative contract | partial | met | not met (S-6) |
| 11 No backend oracle | met | met | **violated** (T-2) |
| 12 No global tolerance | met | met | met |

---

## Appendix — commands and evidence

All commands were run at `9b90ec49deeeb7a099e784301fbf0218a1ddb0e6` with
backends `('python', 'numpy', 'cuda')`, Python 3.14, NumPy 2.5.2, in the
repository's `.venv`.

| Evidence | Command |
| --- | --- |
| Full suite: 1,782 tests, 0 failures, 0 expected failures | `python -m unittest discover -s tests -t .` |
| Arithmetic + power conformance: 241 tests, 0 failures | `python -m unittest tests.operations.arithmetic.test_power_ieee tests.operations.arithmetic.test_power_integer tests.operations.arithmetic.test_power_dtype tests.operations.arithmetic.test_scalar_rounding tests.operations.arithmetic.test_power_accuracy tests.operations.arithmetic.test_power_reference tests.operations.arithmetic.test_power_gradients tests.backend.test_power_execution` |
| Inventory: 60 public operation functions, 6 governed (`sign` forward only) | `tensors.__all__` filtered by `__module__` |
| D-1 reproducer | `ts.abs` / `ts.sign` at `1.401298464324817e-45`, `float32`, three backends |
| D-1 fused/eager split | 20 fusible operations × 2 dtypes × {ordinary, subnormal}, 16,384 elements, fusion asserted reached |
| D-2 reproducer | `2**62 + 1` as `int64` against `1.5` as `float32`, eight operations |
| S-1, S-2 measurements | 4,096 deterministic values, seed 5, ULP against the Python backend |
| S-3 measurements | 22 exceptional cases × 3 backends |
| T-1 counts | `grep -rln use_backend tests/operations/<family>/` |
| T-2 | `tests/backend/_support.py`, `NumPyParityTestCase` |

Every measurement above compares backends **to each other**, which is
deliberately the weaker claim this audit is able to make without the reference
infrastructure of Phase 3. Where a number is quoted as a ULP distance, it is
evidence of the *size* of a divergence, never evidence that any particular
backend is correct.
