# Backend and storage architecture

## 1. Status

> **Status: proposed target contract. None of it is implemented.**
>
> This document states where the backend and storage model is going. It is not
> a description of the package as it stands, and nothing in it may be cited as
> evidence that a behaviour exists. Section 3 records what the package does
> today, measured by running it; section 4 states what is proposed instead;
> every difference between them is unimplemented work.
>
> The implemented contracts remain [Numerical backends](backends.md) for
> selection and execution, [Tensor memory model](memory-model.md) for layout
> and ownership, and [Arithmetic semantics](arithmetic-semantics.md) for
> numerical behaviour. Where this document and those disagree, **they are
> right about today and this one is right about the target.** Neither
> describes the other's subject, so there is one authority per question.
>
> Section 8 separates what is decided from what is not. The backend-selection
> lifecycle (Q1, Q2) is **resolved** in T7 and T8, and construction from a
> `Tensor` (Q3) is **resolved** in 5.1.1. Q4 to Q8 remain open design
> decisions, listed with their alternatives and consequences rather than
> resolved in passing.

## 2. The problem

Three things are distinct in the current implementation, and stating them
precisely matters, because the difficulty is not that the model is ill-defined:

| | What it is |
| --- | --- |
| `Tensor._storage` | The tensor's **authoritative** storage. Its `kind` is the tensor's backend. |
| `Tensor._storage_cache` | Representations of the same values **converted for other backends** and kept for reuse. |
| Backend selection | The backend chosen to **execute** operations — see 2.1, which splits this into the process default and the active backend. |

Asking for a representation does not make it authoritative. `_storage_for(kind)`
returns a cached representation or converts and caches one, and leaves
`_storage` untouched; a tensor used as an operand on another backend keeps the
backend it had. Only `_set_storage` replaces the authoritative storage — it
holds the package's single assignment to `_storage` — and it resets the cache
when it does. The mutation path is what calls it: `_data` reads without
replacing anything, while `_mutable_data` takes the host representation and
installs it, so an in-place write migrates the tensor to the host (3.4).

So a tensor's backend is well defined at every moment, and it is stable unless
one of those specific paths changes it. The cost is elsewhere:

- **A tensor's backend carries no obligation and gives no guarantee.** Knowing
  a tensor is NumPy-backed tells you nothing about where an operation on it
  will run, because the cache will manufacture whatever the active backend
  asks for. The value is well defined and cannot be relied on, which is worse
  than its being undefined: it invites exactly the assumption it does not
  support.
- **The same values are resident in several places at once.** A CUDA tensor
  that has been printed holds both a device buffer and a host copy, and the
  cache has no eviction. Memory cost scales with the number of backends a
  tensor has been used on.
- **Transfers are invisible.** A host-to-device copy costs orders of magnitude
  more than the arithmetic around it, and nothing in the calling code shows
  where one happened.
- **Mismatches are hidden rather than reported.** Combining a host tensor with
  a device tensor silently converts one of them, so a program that has
  accidentally left half its data on the wrong side of the bus still runs, only
  slowly.
- **Cache coherence is a standing obligation.** Every path that changes values
  must go through `_set_storage` so the stale representations are dropped. That
  is currently correct, but it is a rule the whole codebase has to keep, and
  nothing structural enforces it.
- **Every kernel pays for the indirection.** Each one begins by asking for its
  operands in its own representation, because it cannot assume what it was
  given.

The proposal removes the machinery rather than managing it: one representation
per tensor, no cache to keep coherent, no conversion to be invisible, and a
mismatch that is an error.

### 2.1 Process default and active backend

Two selections exist, and the contract below depends on telling them apart.
The implementation already distinguishes them; this document names them.

| Term | What it is | Today |
| --- | --- | --- |
| **Process default** | The backend a process uses when nothing narrower applies. | `_process_backend`, set by `set_backend`, initialized from `TENSORS_BACKEND`. |
| **Active backend** | The backend in force at the point of execution. | `get_backend()`: the context-local override if one is in scope, otherwise the process default. |

`use_backend` sets an override, so it changes the **active backend** without
changing the process default. Because the override is a `ContextVar`, a thread
started inside a scope does not inherit it and uses the process default — as
[backends.md](backends.md#selection) already states.

Wherever this document says an operation runs "under a selection", it means the
**active** backend.

## 3. Existing behaviour

Measured at commit `bba1477` on `python`, `numpy` and `cuda`. This section is
evidence, not contract.

### 3.1 Construction ignores the selection, inconsistently

| built under | `Tensor([…])` | `zeros` / `full` / `eye` | `arange` |
| --- | --- | --- | --- |
| `python` | `PythonStorage` | `PythonStorage` | `PythonStorage` |
| `numpy` | `PythonStorage` | `PythonStorage` | `PythonStorage` |
| `cuda` | `PythonStorage` | `CudaStorage` | `PythonStorage` |

`Tensor([1.0, 2.0])` is host storage under every selection. The creation
operations agree with the selection on CUDA but not on NumPy, and `arange`
does not agree anywhere. A first operation then moves the values:
`t + 1.0` returns `NumPyStorage` under NumPy and `CudaStorage` under CUDA.

### 3.2 A tensor accumulates representations

```
fresh tensor            cache keys: ['python']
after use as an operand cache keys: ['numpy', 'python']
authoritative storage:  PythonStorage
```

The authoritative storage does not change; a second representation is added
beside it and reused by later operations on that backend.

### 3.3 Mixed-backend operands convert silently

With `a` in `PythonStorage` and `b` in `NumPyStorage`:

```
under numpy selection   a + b -> NumPyStorage  [4.0, 5.0]
under python selection  a + b -> PythonStorage [4.0, 5.0]
```

Neither raises. The operand that does not match is converted.

### 3.4 Mutation migrates a tensor to the host

```
before  t[0] = 5.0   NumPyStorage
after   t[0] = 5.0   PythonStorage
```

`_mutable_data` converts to host storage and installs it as authoritative, so
an in-place write moves a device or NumPy tensor to the host permanently.

### 3.5 Construction from a Tensor or Storage keeps the source's backend

```
source                              NumPyStorage
Tensor(src) under python selection  NumPyStorage
Tensor(src) under numpy selection   NumPyStorage
Tensor(src._storage)                NumPyStorage
```

The active backend has no say; the source's representation is copied.

`Tensor.__init__` has two branches for a `Tensor` input, chosen by whether the
requested dtype matches the source's:

```python
if data.dtype == self.dtype:
    self._set_storage(data._logical_storage_for(data._storage.kind).copy())
else:
    self._set_storage(PythonStorage.from_values(data._data, self.dtype))
inferred_shape = data.shape
```

Traced for the three cases that matter, all verified by running them:

| | `a = Tensor([1.0, 2.0], float32)` | `b = Tensor(a)` | `c = Tensor(a, float64)` |
| --- | --- | --- | --- |
| Path | list → `PythonStorage.from_values` | same-dtype branch | different-dtype branch |
| Storage kind | `PythonStorage` under every backend | **the source's kind** | **always `PythonStorage`** |
| Conversion | none | none — `_logical_storage_for` is called with the source's own kind, so it only gathers | host read via `_data`, then per-element `float()` |
| Copy | new buffer | `.copy()` — independent buffer | new buffer |
| Shape | inferred `(2,)` | `data.shape` | `data.shape` |
| Strides / offset | contiguous / `0` | recomputed contiguous / `0` | recomputed contiguous / `0` |
| `_version` | `0` | `0` | `0` |
| Side effect | none | none | populates the source's `python` cache entry |

Two further behaviours, both verified:

- **Layout is normalized.** A non-contiguous source is gathered into logical
  row-major order, and a source at a non-zero offset yields storage holding
  only its logical elements: a `shape=(2,)`, `offset=3` source over a 6-element
  buffer produces a result whose storage size is 2. The result is always
  contiguous with offset 0.
- **No aliasing, in either branch.** `a._storage is b._storage` is false and
  the buffers differ, so mutating `a` leaves `b` and `c` unchanged and does not
  advance their `_version`.

The dtype branch is where the backend is lost. Two constructions differing only
in dtype land on different backends.

`clone()` is currently defined as `Tensor(self)`, so today it takes the
same-dtype branch and returns a tensor on the source's backend — verified: with
the process default `python`, cloning a NumPy-backed tensor yields a
NumPy-backed clone and leaves the active backend alone. That is already what
5.1.2 requires. The delegation nevertheless cannot survive, because the
constructor gains a backend requirement that cloning must not inherit.

The *numerical* rules of the two dtype paths already agree.
`Tensor(a, dtype=uint8)` and `a.astype(uint8)` both raise `OverflowError` on an
out-of-range value, and both give `inf` for a float overflow, exactly as
[arithmetic-semantics §4.5](arithmetic-semantics.md#45-arithmetic-is-not-construction-or-casting)
specifies. Only *where* they run differs.

### 3.6 Display and host access populate the host representation

`repr`, `tolist()` and `item()` work from any representation and leave a
`python` entry in the cache. These are host-facing by definition and their
transfer is intended — see
[backends.md, Storage residency](backends.md#storage-residency-and-transfers).

### 3.7 Graph replay follows the selection at replay time

A graph built under NumPy and replayed under Python produces `PythonStorage`
and the same values. Differentiation behaves likewise: a forward pass under
NumPy differentiated under Python returns host storage. Both work only because
of the implicit conversion in 3.3.

### 3.8 Nested scopes already restore correctly

```
python -> use_backend("numpy") -> numpy -> use_backend("python") -> python
       <- restored numpy <- restored python
```

`use_backend` nests and restores as a context-local override. This behaviour
is kept unchanged by the proposal.

### 3.9 The process default is reassignable at any time

`set_backend` may be called any number of times, with any available backend,
before or after tensors exist. Nothing records that tensors have been built.

Calling it **inside** a scope changes the process default while the override
continues to win, so the change surfaces only on exit:

```
with use_backend("numpy"):
    get_backend()          -> numpy
    set_backend("cuda")
    get_backend()          -> numpy      # the override still wins
    _process_backend       -> cuda       # the default did change
get_backend()              -> cuda       # and surfaces here
```

A scope also restores its previous selection when the block raises, because
`use_backend` resets its token in a `finally`:

```
before        python
inside        numpy
raise RuntimeError
after         python
```

### 3.10 There is no serialization

No `save` or `load` exists in the public API, and `Tensor` is not picklable
(`pickle.dumps` succeeds, `loads` raises `TypeError: Invalid shape`). There is
therefore no serialization behaviour to preserve — see
[question Q6](#q6--serialization).

## 4. The target contract

### T1 — Selection determines construction and execution

The **active backend** (2.1) decides both where a new tensor's values live and
where an operation runs. `Tensor([1.0, 2.0])` with NumPy active produces NumPy
storage; with CUDA active, device storage. Creation operations follow the same rule,
so `zeros`, `full`, `eye`, `arange`, `linspace`, the random samplers and the
initializers all agree with the selection. The inconsistency in 3.1 disappears
because there is one rule.

Construction from an input that **already has** a backend is a separate
question, because there are then two candidate answers. For a `Tensor` input
5.1.1 settles it: the two must agree, or construction raises. For a `Storage`
input it is [Q4](#q4--construction-from-a-storage-object), still open.

### T2 — Each backend uses its own native storage

Python uses `array.array`, NumPy uses `numpy.ndarray`, CUDA uses device-resident
`cupy.ndarray`. This is unchanged and remains an internal detail, not a second
public array API.

### T3 — One authoritative representation

A tensor has exactly one storage. **No alternative representation is retained
to support an implicit transfer.** The storage's `kind` is the tensor's
backend, and it is answerable at any time without qualification.

**A tensor keeps that backend for its whole life.** It is decided by the active
backend when the tensor is constructed (T1) and nothing later changes it:
not a later `set_backend`, not entering or leaving a `use_backend` scope (T8),
not being used as an operand, and not being mutated (5.4). A tensor is never
migrated, because there is nothing to migrate it to.

### T4 — Operations consume matching tensors and produce native storage

An operation runs on the **active backend** (2.1). It requires operands whose
storage belongs to that backend, and produces a result in that backend's
storage. It does not inspect its operands to decide where to run, and it does
not convert them.

### T5 — Cross-backend transfer is outside the execution contract

Combining tensors from different backends, or using a tensor while a different
backend is active, is **not** an operation this contract defines. It is not
slow, or discouraged: it is outside the contract.

### T6 — A mismatch is an error

A mismatch raises, naming the tensor's backend, the active backend and the
operation. It never converts, never falls back, and never succeeds quietly.
The error is the mechanism by which T5 is enforced rather than merely advised.

### T7 — The process default locks on first use

`set_backend` configures the **process default** (2.1). It is chosen once, and
the lock is what makes that choice meaningful.

- **Before the lock**, any available backend may be selected, as often as
  wanted. A program may read configuration, inspect
  `available_backends()` and decide.
- **The lock is taken when the first tensor storage is constructed**, whether
  that happens under the process default or under a scoped override. It marks
  the moment the process stops configuring and starts operating. It is a
  property of the process, not of any tensor, and it is taken once.
- **After the lock, selecting the same backend again is permitted and has no
  effect.** It is idempotent by design, so library or setup code may state its
  requirement without needing to know whether it ran first.
- **After the lock, selecting a different backend raises.** The error names the
  locked default, the requested backend, and that storage has been constructed.

The alternative to raising is not "it works". If the default can change
mid-run, the same unqualified `ts.Tensor([...])` means one backend early in a
process and another backend later, and under T3 and T6 the two results cannot
be combined. The failure surfaces at whichever operation first mixes them, far
from the `set_backend` that caused it. The lock moves it to the call
responsible.

This is not a rule against a process using more than one backend — T8 permits
that deliberately. It is a rule against the *same expression* quietly changing
meaning. A `use_backend` scope states in the source which backend it selects;
a reassigned default changes every later construction that says nothing at all.

**The lock does not depend on how the first tensor was built.** Construction
under a scoped override takes it exactly as construction under the default
does:

```python
ts.set_backend("python")

with ts.use_backend("numpy"):
    a = ts.Tensor([1.0, 2.0])   # takes the lock; a is NumPy-backed

ts.set_backend("cuda")          # raises: the default is locked to python
```

`a` owes nothing to the Python default — it was built under the override, and
T8 guarantees the default never touched it. The lock still applies, because it
is a statement about the **process** and not about `a`: once a process has
started building tensors, the backend that unqualified construction targets is
settled.

Making the lock conditional on a tensor's provenance was considered and
rejected. Its timing would then depend on which scopes happened to be open, so
whether a `set_backend` call succeeded would vary with unrelated code. One
process-wide rule is predictable and states in a sentence when it applies.

**What the lock does not do.** It does not restrict `use_backend` (T8), which
changes the active backend rather than the default. It does not migrate,
convert or invalidate any tensor. It does not make `get_backend()` constant,
because an override can still be in scope.

**`set_backend` inside a scope.** Today this changes the default silently and
the change surfaces when the scope exits (3.9). Under this contract it is
governed by the lock like any other call: before the lock it is permitted and
still surfaces on exit; after the lock it is accepted only for the backend
already locked, and otherwise raises. The scope's override is unaffected
either way.

**Threads.** The default is process-wide, so the lock is too. A thread started
inside a scope does not inherit the override and uses the locked default
(2.1) — which is why the default must be set before workers start.

### T8 — `use_backend` is scoped, exempt, and moves nothing

`use_backend` overrides the **active backend** for the duration of a block. It
is **exempt from T7's lock**: it selects where operations run, not what the
process defaults to, so it may be entered at any time, for any available
backend, however many tensors already exist.

- **It restores the previous selection on exit, including when the block
  raises.** This holds today (3.8, 3.9) and is required, not incidental:
  a scope that leaked its selection on an exception would silently change
  where every later operation ran.
- **Scopes nest**, and each restores the selection it replaced.
- **Entering or leaving a scope never converts, migrates or mutates any
  tensor.** No tensor changes backend because a block was entered, and none
  changes back when it exits.
- **Tensors created inside a scope belong to the scoped backend and outlive
  it.** The scope decided where they were built; it does not own them.
- **Tensors created outside are unaffected**, and under T6 are not usable
  inside a scope selecting a different backend.

The consequence is deliberate: a process may hold tensors of more than one
backend. That is legal under T3 — each has exactly one authoritative
representation — and every attempt to mix them is an error under T6 rather
than a silent conversion. Scoped selection is a supported way to target a
backend, not a loophole the lock is meant to close.

### T9 — Tensor properties are preserved

dtype, shape, strides, offset, ownership, mutation versioning and every rule in
[Arithmetic semantics](arithmetic-semantics.md) are unchanged. This proposal
concerns where values live, not what they are or what operations compute from
them.

That separation is already the stated position:
[§3.4, *Storage residency is not semantics*](arithmetic-semantics.md#34-storage-residency-is-not-semantics)
requires the same values from every backend and defers residency to
[backends.md](backends.md). This proposal changes residency, so it changes
nothing that section governs.

### T10 — Layout handling is not conversion

Gathering a non-contiguous tensor into logical order is a **layout** operation
and happens **within** a backend. It is not a transfer and is not restricted by
T5 or T6. A strided, offset or transposed tensor must still produce correct
logical results on its own backend, and a CUDA tensor's gather happens on the
device.

Conflating the two is the specific mistake this rule exists to prevent: the
current `_logical_storage_for(kind)` performs a conversion and a gather in one
call, so the two questions are answered by one piece of code.

The distinction is already visible in the call sites. `divide`, `power`,
`Tensor.__init__` and `Tensor.contiguous` call it as
`_logical_storage_for(x._storage.kind)` — same kind, so no conversion happens
and only the gather is wanted. `numpy/conversion.py` and `cuda/conversion.py`
call it with a fixed kind, which is where the conversion is. Under the target
contract the first group keeps working unchanged and the second group no
longer needs to ask.

### T11 — No silent fallback

A backend that cannot execute an operation raises
`BackendOperationUnsupportedError`. This is already the contract for `+ - * /`,
`**` and power's gradients
([backends.md, Execution requirements](backends.md#execution-requirements));
the proposal does not weaken it, and T6 adds the matching rule for operands.

## 5. Behaviour under the target model

What follows is required behaviour. It does not prescribe helper functions —
how a kernel obtains its operand is an implementation matter, provided the
behaviour below holds.

### 5.1 Construction and cloning

Construction from a `Tensor` is specified in full in 5.1.1, and cloning — which
is **not** construction and follows a different backend rule — in 5.1.2.

| input | required behaviour |
| --- | --- |
| Python scalar | storage for the active backend, shape `()` |
| nested list | flattened and stored on the active backend |
| `array.array` | values read on the host, stored on the active backend |
| **`Tensor`** | backend must match the active backend; see 5.1.1 ([Q3](#q3--construction-from-a-tensor--resolved), resolved) |
| **`Storage`** | see [Q4](#q4--construction-from-a-storage-object) — still open |

The scalar, list and `array` cases are unambiguous: they are host inputs with
no backend of their own, so they take the active backend's. A `Tensor` input
already has a backend, which 5.1.1 settles. A `Storage` input also already has
one, and Q4 is deliberately left open.

Everything in this table is **construction**, and construction answers to the
active backend. `source.clone()` is not in the table because it is not
construction: it answers to the source's backend (5.1.2).

#### 5.1.1 Construction from a Tensor

`Tensor(source)` and `Tensor(source, dtype=…)` produce a **new, independent
tensor holding the source's logical values**.

This section governs the **constructor only**. `source.clone()` is an
operation on an existing tensor, not a construction, and 5.1.2 governs it
instead. The two differ in exactly one respect — which backend they answer to —
and in nothing else.

**R1 — The source's backend must be the active backend.** If it is not,
construction raises the mismatch error of T6, naming the source's backend, the
active backend and the constructor. It does not convert, does not copy the
values to the host, and does not adopt the source's backend in defiance of the
active one.

```python
with ts.use_backend("numpy"):
    a = ts.Tensor([1.0, 2.0])

with ts.use_backend("python"):
    b = ts.Tensor(a)        # raises: source is numpy, active backend is python
```

**R2 — The result is on that same backend.** Since the source's backend and
the active backend are equal whenever construction succeeds, there is only one
answer and no choice to make. Construction is never a transfer (T5).

**R3 — The result is an independent deep copy.** It owns storage no other
tensor references. This is not new: it is
[memory-model.md's existing guarantee](memory-model.md#ownership-and-current-limits)
that public construction copies and introduces no shared-storage aliasing, and
this contract keeps it unchanged.

**R4 — The result's layout is normalized.** Its shape is the source's shape,
its strides are contiguous for that shape, and its offset is zero. Its storage
holds exactly `shape.size` elements in logical row-major order.

**R5 — A non-contiguous or offset source is gathered on its own backend.**
Producing logical row-major order from strides and an offset is a layout
operation, not a transfer (T10). A CUDA source is gathered on the device. The
resulting values are the source's logical values, in logical order.

**R6 — Without an explicit dtype, the source's dtype is preserved.** With an
explicit dtype, conversion happens **on the active backend**, and the result is
that backend's native storage. The converted values are exactly what the
current host path produces; see R7.

**R7 — dtype conversion changes residency, not numbers.** The numerical rules
are those already specified in
[arithmetic-semantics §4.5](arithmetic-semantics.md#45-arithmetic-is-not-construction-or-casting):
an out-of-range integer target raises, float overflow yields `inf`, and
float→integer truncates toward zero. This contract does not alter them, add a
tolerance, or introduce a second casting rule. What changes is only that the
conversion stops moving the tensor to host storage.

**R8 — Mutation of either tensor never affects the other.** It follows from
R3: with no shared storage there is no propagation. The result's `_version`
starts at zero and is independent of the source's, so mutating the result
cannot invalidate a graph that recorded the source, and mutating the source
cannot invalidate one that recorded the result.

**R9 — Construction records nothing in the graph and propagates no gradient.**
`Tensor` is the non-differentiable value type; gradient tracking belongs to
`Variable`, and `Tensor(variable)` raises `TypeError` today and continues to.
Constructing a tensor inside a traced region adds no node and creates no edge,
so **this contract requires no change to the graph or differentiation
contract**. A `Variable` wrapping a tensor still aliases that tensor rather
than copying it, which is `Variable`'s existing behaviour and is untouched
here: R3 is a guarantee about tensor-to-tensor construction, not about what
`Variable` does with a tensor afterwards.

##### Behavioural matrix

`A` is the active backend. Every row assumes construction succeeds unless it
says otherwise.

| # | Case | Result backend | Result dtype | Storage | Layout | Values |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Same backend, dtype unchanged | `A` | source's dtype | new, independent | contiguous, offset 0, size = `shape.size` | source's logical values, unchanged |
| 2 | Same backend, dtype explicitly different | `A` | the requested dtype | new, independent, **native to `A`** | as row 1 | converted per §4.5; raises on out-of-range integer targets |
| 3 | Source backend ≠ `A` | — | — | — | — | **raises** the T6 mismatch error; nothing is constructed or converted |
| 4 | Non-contiguous or offset source, same backend | `A` | as rows 1–2 | new, independent, compacted | contiguous, offset 0 | gathered on `A` into logical row-major order (T10) |
| 5 | Mutation after construction | unchanged | unchanged | disjoint from the source's | unchanged | neither tensor observes the other's writes; `_version` counters are independent |

Row 3 is the resolution of Q3. Rows 1, 2, 4 and 5 hold whenever row 3 does not
apply.

##### What changes from today

| | Today (3.5) | Under this contract |
| --- | --- | --- |
| Cross-backend construction | Silently keeps the source's backend | Raises (R1) |
| dtype-changing construction | Always produces `PythonStorage` | Produces the active backend's storage (R6) |
| Same-backend, same-dtype | Independent copy, layout normalized | **Unchanged** (R3, R4) |
| Converted values | Per §4.5 | **Unchanged** (R7) |
| Graph and gradients | Not involved | **Unchanged** (R9) |

Two of the five rows are already correct, which is the point: this resolution
removes two behaviours and preserves everything else.

##### Dependencies discovered while resolving this

Recorded because an implementer will meet them, not decided here.

- **The dtype path depends on the cast dispatching strictly.** R6 requires a
  dtype-changing construction to produce the active backend's storage. Today
  the cast path, `execute_cast`, consults the workload policy and runs the
  Python reference below a size threshold: verified, a 4-element
  `astype(float64)` under NumPy returns `PythonStorage` while a 5000-element
  one returns `NumPyStorage`. So R6 cannot be satisfied by routing
  construction through the cast as it currently dispatches. This is **not a
  new decision** — T4 already requires any operation to produce the active
  backend's storage, and [backends.md](backends.md#execution-requirements)
  already records that operations other than `+ - * /` and `**` still follow
  the workload policy. R6 simply makes the gap observable at a second call
  site.
- **Q4 is genuinely separate.** Resolving Q3 does not resolve construction
  from a `Storage`, for the reasons given under
  [Q4](#q4--construction-from-a-storage-object). An implementation of 5.1.1
  must not quietly settle it by sharing a code path.
- **Q8 becomes load-bearing.** With R1 in force there is no way to move a
  tensor between backends. That is intended, and
  [Q8](#q8--an-explicit-migration-api) remains open; no migration API is
  introduced here.
- **No conflict with the existing ownership contract.** R3 and R8 restate
  guarantees [memory-model.md](memory-model.md#ownership-and-current-limits)
  already makes. Nothing in this resolution weakens or extends them, and the
  `Variable`-aliases-its-tensor behaviour noted in R9 is likewise untouched.
- **No change to the graph contract.** R9 is a statement of existing
  behaviour, confirmed by `Tensor(variable)` raising `TypeError` and by the
  construction path never touching graph state.

#### 5.1.2 Cloning

`source.clone()` returns a new, independent tensor with the source's logical
values, shape and dtype. It is **an operation on an existing tensor, not a
construction**, and the difference is deliberate.

**C1 — The clone is on the source's backend, whatever the active backend is.**
Cloning answers to the tensor it is called on, not to the selection in force.
It is therefore **exempt from the constructor's R1** and never raises a
mismatch.

```python
ts.set_backend("python")

with ts.use_backend("numpy"):
    a = ts.Tensor([1.0, 2.0])   # a is NumPy-backed

b = a.clone()                   # b is NumPy-backed; the default is still python
```

**C2 — No transfer occurs, in either direction.** The clone does not move to
the active backend, does not build a host-backed intermediate on the way, and
does not populate any alternative representation of the source. Reading a
tensor's own values on its own backend is not a transfer (T10), and there is
nothing here that crosses a backend boundary at all.

**C3 — Cloning never changes the active backend or the process default.** It
selects nothing. The example above leaves the process default at `python`
throughout.

**C4 — Every copying invariant of 5.1.1 applies unchanged.** Independent deep
copy owning storage nothing else references (R3); shape preserved, strides
contiguous, offset zero, storage exactly `shape.size` elements (R4); a
non-contiguous or offset source gathered **on its own backend** into logical
row-major order (R5); dtype preserved (R6, with no dtype parameter to change
it); mutation of either tensor invisible to the other, the clone's
mutation-version counter starting at zero (R8); and no graph node, edge or
gradient participation (R9).

**C5 — `clone()` remains the only public spelling of this operation.** No new
API is introduced, and no parameter is added to it. A tensor is cloned onto
its own backend or not at all.

**This section changes no behaviour.** Verified against the current
implementation: with the process default `python` and a NumPy-backed `a`,
`a.clone()` already returns a NumPy-backed tensor with independent storage, an
untouched active backend, no new entry in the source's representation cache,
and a version counter of zero — C1 through C4 exactly. What the target contract
changes is only that `clone()` can no longer be *expressed* as `Tensor(self)`,
because the constructor will raise where cloning must succeed. The behaviour
being specified here is the behaviour that exists; an implementation may be
tested against it before and after the refactor and must produce identical
results.

##### `Tensor(source)` against `source.clone()`

| | `Tensor(source)` | `source.clone()` |
| --- | --- | --- |
| Kind of thing | Constructor (5.1.1) | Operation on an existing tensor (5.1.2) |
| Backend it answers to | The **active** backend | The **source's** backend |
| Source backend ≠ active backend | **Raises** (R1) | **Succeeds**, on the source's backend (C1) |
| Result backend | The active backend | The source's backend |
| dtype parameter | Accepted; conversion on the active backend (R6, R7) | None |
| Storage, layout, mutation, graph | R3, R4, R5, R8, R9 | Identical (C4) |

The distinction is narrow on purpose. Only the backend question differs;
everything about what the copy *is* stays the same. The reason they differ at
all is that they ask different questions: `Tensor(x)` asks "build me a tensor
here", where *here* is the active backend, and `x.clone()` asks "give me
another one of these", where *these* already has a backend and no selection
is involved.

### 5.2 Nested backend contexts

Scoped selection behaves as it does today (3.8, 3.9) and T8 requires that it
keep doing so. Stated as behaviour:

| Situation | Required behaviour |
| --- | --- |
| Entering a scope | The active backend becomes the scoped one. The process default is unchanged. No tensor is touched. |
| Nesting scopes | Each scope restores the selection it replaced, innermost first. |
| Leaving a scope normally | The previous active backend is restored. No tensor is touched. |
| Leaving a scope by exception | Identical to leaving normally. The selection is restored before the exception propagates. |
| A tensor built inside a scope | Belongs to the scoped backend, and still does after the scope exits. If it is the process's first, it takes T7's lock like any other first construction. |
| A tensor built outside a scope | Unchanged by the scope. Not usable inside one selecting a different backend (T6). |
| A thread started inside a scope | Does not inherit the override; uses the locked process default (2.1, T7). |
| `set_backend` called inside a scope | Governed by T7's lock, not by the scope. The override continues to win until the scope exits. |

A tensor's backend is the **active** backend at the moment it was constructed,
and nothing about entering or leaving a scope changes it afterwards. A program
that constructs under `use_backend("cuda")` and then reads `tolist()` outside
the scope still reads a device tensor's values, because host-facing access
(5.5) does not depend on the selection at all.

### 5.3 Backend mismatch

A mismatch is detected where the operand is consumed and raises there. The
error names the tensor's backend, the active backend and the operation, so the
cause is legible without a debugger. Three cases:

- **Operand against the active backend.** A NumPy tensor used while CUDA is
  active.
- **Operand against operand.** A NumPy tensor and a CUDA tensor in one
  operation, whatever the selection.
- **Operand against a graph.** A recorded computation replayed with operands
  from a different backend — see [Q5](#q5--graph-replay-across-backends).
- **Source against the active backend.** `Tensor(source)` where the source's
  backend is not the active one (5.1.1, R1). Construction is not exempt from
  T6 merely because it produces a new tensor rather than consuming two.

`source.clone()` is **not** in this list. It names no backend and consumes no
second operand, so there are never two answers to disagree: it operates on the
source's backend by definition (5.1.2, C1).

### 5.4 Mutation

An in-place write updates the tensor's own storage, on its own backend. It does
**not** migrate the tensor to the host, which is the behaviour in 3.4. Mutation
versioning and stale-autograd detection are unchanged (T9).

A write whose value comes from the host — `t[0] = 5.0` — transfers one scalar
to the tensor's backend. That is a host-facing write, the mirror of a
host-facing read (5.5), and is intended.

### 5.5 Display and host access

`repr`, `str`, `tolist()`, `item()` and comparison to a Python value read
values on the host. **This is the caller's request, not an implicit transfer**,
and it remains permitted from any backend under any selection. It is the same
position [backends.md](backends.md#storage-residency-and-transfers) already
takes.

The one change from 3.6: the host values obtained this way are **not retained**
on the tensor. There is no cache for them to populate, so a second `tolist()`
transfers again. [Question Q7](#q7--repeated-host-reads) records whether that
cost is acceptable.

### 5.6 Serialization

There is nothing to preserve (3.10), so no behaviour is specified. If
serialization is added, whether a stored tensor records its backend is
[question Q6](#q6--serialization).

### 5.7 Graph execution

A recorded computation describes operations and operands, not a backend. Under
the target model its inputs have backends, so replay is constrained by them
rather than by where the graph was built. Replay under a selection matching
its inputs produces results on that backend, as it does today (3.7); replay
against a different one is a mismatch under 5.3, which is a behaviour change
from 3.7 and is [question Q5](#q5--graph-replay-across-backends).

### 5.8 Differentiation

A gradient is produced on the backend of the tensors it is computed from, and
the VJP kernels are subject to T4 and T6 like any other operation. The
numerical contract — the region tables, accuracy bounds and classification
rules of [Arithmetic semantics §12.7](arithmetic-semantics.md#127-differentiation-d7)
— is untouched (T9).

The forward-under-one-backend, backward-under-another case in 3.7 becomes a
mismatch. That is the intended consequence of T5: it works today only through
the implicit conversion that this proposal removes.

## 6. Affected modules

Listed as the surface the refactor touches, with what the target contract
requires of each. **No replacement helper is named**, deliberately: the
objective is to remove indirection, not to rename it.

| Module or member | Today | Required |
| --- | --- | --- |
| `tensors/tensor.py` — `_storage_cache` | Maps kind to representation; populated on demand | **Gone.** T3 leaves one representation, so there is nothing to cache. |
| `tensors/tensor.py` — `_storage_for(kind)` | Converts and caches | **Gone.** Asking for a tensor in another backend is the request T5 removes. |
| `tensors/tensor.py` — `_logical_storage_for(kind)` | Converts, then gathers | **Split.** The gather survives as a within-backend layout operation (T10); the conversion does not. |
| `tensors/tensor.py` — `_set_storage` | Installs storage, resets the cache | Installs storage. The dtype agreement check it also performs is unrelated and stays. |
| `tensors/tensor.py` — `_mutable_data` | Converts to host and installs it | **Changed.** Mutation acts on the tensor's own backend (5.4). |
| `tensors/tensor.py` — `_data` | Gathers host values | Becomes a host-facing read (5.5), not a conversion. |
| `tensors/tensor.py` — `__init__` | Host storage for scalars/lists; copies a Tensor's or Storage's kind; a dtype change routes through host values | **Changed.** T1 for host inputs; 5.1.1 for a `Tensor` input, whose two dtype branches collapse into one backend-native path; [Q4](#q4--construction-from-a-storage-object) still open for a `Storage` input. |
| `tensors/tensor.py` — `clone` | Defined as `Tensor(self)`; already returns the source's backend whatever the active backend is | **Behaviour unchanged; the definition must change.** 5.1.2 requires exactly what `clone()` already does, but `Tensor(self)` stops expressing it once the constructor gains R1, because it would then raise whenever the source's backend and the active backend differ. |
| `tensors/backend/dispatch/manipulation/cast.py` — `execute_cast` | Falls back to the Python reference below a workload threshold | **Changed.** 5.1.1 R6 requires a dtype conversion to produce the active backend's storage, which T4 already requires of any operation. See the dependency note in 5.1.1. |
| `tensors/backend/conversion.py` — `convert_storage` | The only cross-backend conversion | **Gone**, unless a deliberate migration API is added — see [Q8](#q8--an-explicit-migration-api). Its one caller is `_storage_for`. |
| `tensors/backend/numpy/conversion.py` — `_view`, `_operand`, `_arithmetic_operand` | Each begins by asking for a NumPy representation | **Simplified.** Under T4 the operand already is one; what remains is dtype handling and the logical gather. |
| `tensors/backend/cuda/conversion.py` — the same three, plus `_widen`, `_narrow`, `_working_values` | As above, plus the PTX-based binary32 widening | As above. The subnormal-preserving widening is numerical (T9) and stays exactly as it is. |
| `tensors/backend/config.py` — `set_backend` | Reassignable at any time (3.9) | **Changed.** Sets the process default and locks it on first tensor storage; same-backend calls are accepted, a different backend raises (T7). |
| `tensors/backend/config.py` — `use_backend` | Scoped context-local override, restored in a `finally` | **Unchanged**, and exempt from the lock. T8 makes the existing restore-on-exception behaviour a requirement rather than an implementation detail. |
| `tensors/backend/config.py` — `get_backend` | Override if in scope, else the process default | **Unchanged.** It already answers the active-backend question (2.1). |
| `tensors/backend/policy.py` | Workload thresholds choose a path | Untouched by this proposal; still governed by [backends.md](backends.md#execution-requirements). |
| Creation ops and `tensors/init/` | Mixed agreement with the selection (3.1) | **Changed.** T1 makes them uniform. |
| `tensors/graph/` | Replays under the current selection | **Changed** at the boundary only: operands are checked (5.7). Recording, replay order and differentiation are untouched. |

## 7. What this proposal does not change

- Numerical semantics of any operation, including dtype promotion, scalar
  conversion, exceptional values, subnormals and accuracy (T9).
- Layout: `Shape`, `Strides`, `offset`, contiguity, and the results a
  non-contiguous tensor produces (T10).
- Ownership and copying guarantees, and the deferred view work in
  [memory-model.md](memory-model.md#ownership-and-current-limits).
- Execution-location requirements and the absence of the observability API
  ([backends.md](backends.md#observability)).
- Kernel coverage: which operations each backend implements.

## 8. Questions

Q1, Q2 and Q3 are **resolved**. Each decision is normative elsewhere — T7 and
T8 for the lifecycle, 5.1.1 for construction from a `Tensor` — and the entries
here are kept so the rejected alternatives stay on record. Q4 to Q8 remain
**open**: each records the alternatives, what follows from them, and a
recommendation, and none of those recommendations has been adopted.

### Q1 — When the process default locks — RESOLVED

*Resolved in [T7](#t7--the-process-default-locks-on-first-use): the lock is
taken when the first tensor storage is constructed, and applies to the process
default only.*

What marks a process as locked, and what is the unit the rule applies to?

| Option | Consequence |
| --- | --- |
| **A. First tensor storage constructed — ADOPTED** | A program may select freely during import and configuration, and is fixed from its first tensor — built under the default or under a scoped override alike. Requires one process-wide flag and no provenance tracking. |
| **B. First operation executed** | More permissive: tensors may be built, then the backend chosen. But tensors built before the choice already have a backend, so T3 makes them mismatched — the rule would not prevent the failure it exists to prevent. |
| **C. Per-thread rather than per-process** | Allows a worker per backend. Multiplies the lifecycle by the thread model and interacts with `use_backend`'s context-local override in ways not thought through. |

**Adopted: A.** B does not achieve the rule's purpose. C was not adopted
because the process default is process-wide and making it per-thread is a
separate design; a process that wants to use two backends does so with
`use_backend` (T8), which A leaves fully available.

Resetting the lock is a test-support concern, not part of the public contract.
Since the suite exercises three backends in one process through `use_backend`,
which T8 exempts, the need for a reset hook is an implementation question and
is deliberately not specified here.

### Q2 — `use_backend` against the lock — RESOLVED

*Resolved in [T8](#t8--use_backend-is-scoped-exempt-and-moves-nothing):
`use_backend` is exempt.*

T7 locks the process default; T8 lets a block select a different backend. The
two had to be reconciled.

| Option | Consequence |
| --- | --- |
| **A. `use_backend` is exempt — ADOPTED** | Scoped selection keeps working. A block may construct tensors on a second backend, so one process holds tensors of two backends — legal under T3, and every cross-use is an error under T6. |
| **B. `use_backend` is restricted to the locked backend** | The lock holds absolutely; `use_backend` becomes an assertion rather than a selection, and its remaining purpose is unclear. |
| **C. `use_backend` is removed** | Simplest model, largest breaking change. The test suite uses it heavily to exercise all three backends in one process. |

**Adopted: A.** The lock exists to stop a program's tensors silently
disagreeing with the default they were built under; a scoped block that
constructs and consumes its own tensors does not create that problem, and T6
catches it if one escapes. B and C would remove a legitimate capability to
enforce a rule that was never aimed at it.

The distinction this turns on is the one drawn in 2.1: T7 governs the
**process default**, T8 governs the **active backend**. They are different
selections, so exempting one from the other's lock is not an exception to the
model — it is the model.

### Q3 — Construction from a Tensor — RESOLVED

*Resolved in [5.1.1](#511-construction-from-a-tensor), which is normative. The
entry is kept so the rejected alternatives stay on record.*

`Tensor(other)` where `other` belongs to a different backend.

| Option | Consequence |
| --- | --- |
| **A. Raise — ADOPTED** | Consistent with T6: construction is not a transfer mechanism. The caller has no way to move data between backends, which is the intended consequence, not an oversight. |
| **B. Copy to the active backend** | Convenient, and a natural reading of "construct a tensor here from these values". But it is a cross-backend transfer under an ordinary-looking constructor, which is the pattern T5 removes. |
| **C. Copy the source's backend** | Today's behaviour (3.5). Construction then ignores the active backend, contradicting T1. |

**Adopted: A.** B hides the same transfer behind a different spelling; C
contradicts T1. Under A there is exactly one way a tensor acquires a backend —
the active backend at its construction — and no expression quietly moves data
across a bus.

Same-backend construction was specified at the same time, because resolving
only the cross-backend half would have left the more common case undefined.
5.1.1 states both, and its matrix row 3 is this decision.

**Cloning is not covered by A.** `source.clone()` asks a different question and
gets a different answer: it produces a tensor on the **source's** backend
whatever the active backend is, and never raises a mismatch. That is 5.1.2, and
it is the reason `clone()` cannot remain defined as `Tensor(self)`. A applies
to the constructor, not to every operation that copies a tensor.
**Relationship to Q8.** A leaves no way to move a tensor between backends, so
it is A that gives Q8 its force. Q8 remains open and is **not** decided here;
if a migration API is ever added it will be an explicit, separately named
operation, never this constructor. No migration API is introduced by this
resolution.

### Q4 — Construction from a Storage object

`Tensor(storage)` where the storage belongs to a different backend. The same
three options as Q3 apply, with one difference: a `Storage` is an internal
type, so the public surface is narrower and the compatibility cost of raising
is lower.

**Recommended: A (raise)**, for consistency with Q3. **Not adopted** — Q3's
resolution does not decide this one. The two are separate because a `Storage`
has no shape, strides or offset of its own, so the layout and gather rules of
5.1.1 (R4, R5) have no counterpart here, and `Tensor(Storage)` carries the
additional ownership guarantee recorded in
[memory-model.md](memory-model.md#ownership-and-current-limits). Resolving Q3
constrains the answer but does not supply it.

### Q5 — Graph replay across backends

A computation recorded under one backend, replayed under another.

| Option | Consequence |
| --- | --- |
| **A. Mismatch, raises** | Follows from T6 with no special case. Today's behaviour (3.7) breaks, including the forward-here-backward-there pattern in 5.8. |
| **B. Graphs carry a backend** | Replay checks against the recorded backend rather than the current selection. Adds state to the graph and a second place where a backend is remembered. |
| **C. Replay rebuilds inputs on the active backend** | Preserves today's behaviour by reintroducing exactly the implicit transfer T5 removes. |

**Recommended: A.** B and C both restore the ambiguity elsewhere. The cost is
that a legitimate pattern — measure on one backend, differentiate on another —
stops working, and that should be confirmed as acceptable before implementing.

### Q6 — Serialization

No serialization exists (3.10), so this is a question about future work.

| Option | Consequence |
| --- | --- |
| **A. Serialize values and dtype only** | A loaded tensor takes the active backend, like any host input (T1). Portable across machines with different hardware. |
| **B. Serialize the backend too** | Round-trips exactly. A file written on a CUDA machine cannot be loaded on one without a device, unless loading falls back — which T11 forbids. |

**Recommended: A** if serialization is added. Nothing needs deciding until it
is.

### Q7 — Repeated host reads

T3 removes the cache, so `tolist()` twice transfers twice (5.5).

| Option | Consequence |
| --- | --- |
| **A. Accept the cost** | Simple and honest: each host read is a transfer, and the caller can see how many they asked for. Printing a large device tensor in a loop is slow. |
| **B. Cache host values** | Restores a second representation, which is what T3 removes, and raises a staleness question on mutation. |

**Recommended: A.** B is the cache under another name. Whether the cost bites
in practice should be measured once the rest is implemented.

### Q8 — An explicit migration API

With implicit transfer gone, there may be no way to move a tensor between
backends. Whether one is wanted is a product question, not a structural one.

| Option | Consequence |
| --- | --- |
| **A. None** | The strictest model. A program uses one backend, or constructs each tensor under the selection it belongs to. Q3 option A becomes hard to work around, which may be right. |
| **B. An explicit method** | Transfers become visible and greppable, which is the property the current model lacks. Adds one public API, and the "do not invent additional conversion APIs" instruction means it needs a stated requirement first. |

**Recommended: defer.** No concrete requirement for it has been stated. Q3 is
now resolved as A, so this question is live rather than hypothetical: there is
currently no way to move a tensor between backends, by design. Decide once the
implementation shows whether the strict model is usable in practice; adding a
migration API later is compatible, removing one later is not.

## Related documents

- [Numerical backends](backends.md) — selection, execution requirements and
  residency **as implemented**.
- [Tensor memory model](memory-model.md) — layout, ownership and the deferred
  view work.
- [Arithmetic semantics](arithmetic-semantics.md) — the numerical contract,
  unaffected by this proposal.
- [Package structure](package-structure.md) — where the affected modules live.
