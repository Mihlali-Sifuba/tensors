# Backend and storage architecture

## 1. Status

> **Status: proposed target contract, with one section implemented.**
>
> This document states where the backend and storage model is going. Apart
> from 5.4, it is not a description of the package as it stands, and nothing
> in it may be cited as evidence that a behaviour exists. Section 3 records
> what the package does today, measured by running it; section 4 states what
> is proposed instead; every difference between them is unimplemented work.
>
> **5.4, mutation, is implemented**, and 3.4 records it among the measured
> behaviours. It was separable from the rest because it asks only where a
> tensor already lives, never where it should live, so it needed neither the
> selection lifecycle of T7 nor the single-representation rule of T3. No
> other section of 5 may be read as implemented on the strength of it.
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
> `Tensor` (Q3) is **resolved** in 5.1.1, and construction from a `Storage`
> (Q4) is **resolved** in 5.1.3. Graph replay (Q5) is **resolved** in
> 5.7 and 5.8, host access (Q7) is **resolved** in 5.5, and explicit migration
> (Q8) is **resolved** in 5.9. Only serialization (Q6) remains open and
> deferred.

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
when it does. The mutation path used to call it: `_data` reads without
replacing anything, while `_mutable_data` took the host representation and
installed it, so an in-place write migrated the tensor to the host. That path
is gone, and mutation now writes the authoritative storage where it already
lives (3.4).

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

### 3.4 Mutation preserves a tensor's backend — implemented

> **This is the one part of section 5 that is implemented.** It is recorded
> here, among the measured behaviours, because it now *is* the measured
> behaviour. Everything else in section 5 remains proposed.

```
before  t[0] = 5.0   NumPyStorage
after   t[0] = 5.0   NumPyStorage
```

An in-place write updates the tensor's own storage, on its own backend, as
5.4 requires. `_mutable_data` — which converted to host storage and installed
it as authoritative, so that a single element write moved a NumPy or device
tensor to the host permanently — no longer exists. The write is dispatched to
the backend the destination storage already belongs to, and only the cache of
representations converted from it is discarded afterwards, leaving the
authoritative storage in place.

A value that comes from the host still reaches the tensor's backend: `t[0] =
5.0` transfers that one scalar, which is the host-facing write 5.4 allows.
The tensor is not transferred in the other direction, and a tensor supplying
the values is not read out to the host to supply them.

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

The `Storage` branch is shorter but has a different contract. It calls
`data.copy()` directly, infers `(data.size,)`, then applies an explicit shape
if one was supplied. The result is contiguous, has offset zero and starts at
version zero. The copy is independent: neither the `Storage` object nor its
mutable backend buffer is shared. Mutation of a tensor already using the
source storage therefore cannot affect the result, and mutation of the result
cannot affect the source.

That branch also exposes two behaviours verified separately at `bbd95b8`:

- it never reads the active backend, so a storage whose `kind` differs from
  the selection is copied successfully and the result keeps the storage's
  kind; and
- an explicit dtype different from `storage.dtype` raises `TypeError` in
  `_set_storage`. No conversion is attempted. An equal explicit dtype is
  accepted.

With no explicit shape, two stored elements become shape `(2,)`. With an
explicit shape such as `(2, 1)`, only the element count is carried over:
strides are recomputed as contiguous and offset is zero. A shape whose size
does not equal `storage.size` raises `ValueError`. This is not tensor layout
normalization: `Storage` is flat and carries no shape, strides or offset to
preserve or gather.

The private construction paths are intentionally different:

- `_from_owned_storage` installs the exact `Storage` object and buffer without
  copying, checks an optional expected dtype and the requested shape's element
  count, and initializes contiguous strides, offset zero and version zero. It
  does not inspect the active backend. Its callers include creation and random
  functions, eager forward and backward operations, slicing and casting,
  graph fusion, and optimizer updates; the supplied storage is the newly
  produced result whose ownership those callers transfer.
- `_from_metadata` copies its supplied storage because it is the internal
  layout-construction path, not an ownership transfer. `_from_values` creates
  fresh `PythonStorage`, and `_replace_owned_storage` adopts a same-size,
  same-dtype internal result while advancing an existing tensor's version.

These paths all accept or produce `Storage`, but they do not promise the same
ownership. In particular, `_from_owned_storage` cannot safely replace public
`Tensor(Storage)`: if its caller retains and mutates the supplied buffer, the
returned tensor observes that mutation. Exclusivity is a caller precondition,
not something the current implementation can prove.

### 3.6 Display and host access populate the host representation

`repr` (also used by `str`, because no separate `__str__` is defined),
`tolist()`, `item()`, scalar indexing, scalar formatting and tensor equality
ultimately read `_data`. `_data` calls
`_logical_storage_for("python")`, which first calls `_storage_for("python")`.
For NumPy storage this serializes the native buffer through bytes into an
`array.array`; for CUDA storage it first calls `cupy.asnumpy` and then takes
the same bytes path. `_storage_for` retains that full physical
`PythonStorage` in `_storage_cache`. A non-compact tensor is gathered into a
second, compact `PythonStorage` in logical row-major order; that compact
object is temporary, but the full physical host representation remains
cached. A compact Python-backed tensor simply exposes its authoritative
buffer to `_data`.

`tolist()` copies those logical values into an independent flat list and
`item()` returns one Python scalar. `repr` repeatedly indexes `_data` while it
formats nested display text. Complete integer indexing is different from a
slice: it first runs the selected backend's slice operation to produce a
zero-dimensional tensor, then calls `item()` on that result. Tensor equality
compares shapes and then two `tolist()` results; elementwise comparison
operations still produce tensors and are not host observations.

The intended transfer for these public host-facing operations is consistent
with [backends.md, Storage residency](backends.md#storage-residency-and-transfers).
The current helper is much broader, however: 109 Python reference-kernel
files and many operation, gradient, validation, broadcasting and manipulation
helpers also read `_data`. Those internal reads can silently populate host
storage while numerical work is executing, so they are not evidence that
public inspection and internal execution have the same permission.

### 3.7 Graph replay follows the selection at replay time

A graph built under NumPy and replayed under Python produces `PythonStorage`
and the same values. Differentiation behaves likewise: a forward pass under
NumPy differentiated under Python returns host storage. Both work only because
of the implicit conversion in 3.3.

The absence of a backend field on a graph is real, but it is not the whole
execution model. Tracing and structural recording create `VariableNode` and
`OperationNode` identities joined by ordered edges. Compilation turns those
identities into value slots and immutable `Instruction` objects holding an
operation plus input and output slot numbers. Neither representation stores a
backend, kernel object or tensor buffer.

Residency enters through the values bound to leaf slots:

- replay inputs are `Tensor` objects wrapped or rebound to leaf `Variable`s;
- model parameters and captured `Variable`s remain bound leaves;
- captured `Tensor` constants are wrapped as non-gradient bound leaves; and
- scalar operands in an eager trace become zero-dimensional, non-gradient
  tensor leaves. A structural expression cannot type a bare scalar before
  values exist, so it falls back to tracing rather than storing an untyped
  literal in the structural graph.

`Computation.forward()` allocates a fresh local slot buffer, seeds its leaf
slots from those Variables, and calls each recorded operation's `forward` in
dependency order. Results replace the data of the already-bound result
Variables (or materialize them on the first pass), and each result captures
operand/result mutation state for differentiation. Result storage is produced
by ordinary backend dispatch and adopted through `_from_owned_storage`; the
graph layer does not choose a storage kind.

Three replay mechanisms differ in their preparation but converge on that
same `Computation.forward()` path:

- a structurally built model records and compiles once, then rebinds call
  `Tensor`s to its input vertices;
- an explicitly compiled trace stores input-to-Variable bindings, output and
  computation references, structural metadata and leaf shape/dtype/gradient
  guards; and
- direct `Computation.forward()` reads whatever Tensors its bound leaf
  Variables currently hold.

The guarded trace's call signature currently includes `get_backend()`. A
backend change therefore causes a cache miss and retrace even though the
instruction sequence is backend-neutral. Structural replay and direct
`Computation` replay have no equivalent backend guard. All paths still work
cross-backend today because kernels obtain converted representations through
the tensor storage cache.

Fusion does not make the graph itself backend-specific. `_FusionPlan` caches
shape/dtype-derived ranges and backend-neutral step descriptions beside the
canonical instructions. Dispatch asks the backend active for that pass to
execute them or runs the same instructions ordinarily. CUDA `RawKernel`
objects are compiled and cached inside the CUDA kernel module, keyed by the
typed expression and shapes; the graph does not own them. Kernel lookup is
selection-aware and its cache is cleared when selection changes.

Backward first validates saved forward mutation states, builds or validates an
upstream seed, and traverses the same instructions in reverse. VJPs and fused
VJPs dispatch under the backend active for the reverse pass; gradients are
constructed from returned native storage. `backward()` publishes `.grad`
values only after the complete reverse pass succeeds, while `grad()` returns
from local gradient buffers without mutating `.grad`. Higher-order
differentiation records the VJP operations as a new graph.

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

### 3.11 Conversion exists only as cache machinery

There is no `Tensor.to` method or other public migration API. The sole generic
cross-backend primitive is `convert_storage`, called only by
`Tensor._storage_for(kind)`, which retains its result in `_storage_cache`.
Same-kind conversion returns the original `Storage`, not a copy.

The cross-kind branches are pair-specific: NumPy to Python serializes through
bytes; CUDA to Python uses `cupy.asnumpy` and then the bytes path; Python to
NumPy uses `numpy.frombuffer`; CUDA to NumPy uses `cupy.asnumpy`; Python to
CUDA stages through a NumPy `frombuffer` view before `cupy.asarray`; and NumPy
to CUDA uses `cupy.asarray`. The Python-to-NumPy result shares memory with the
source Python buffer, while the other cross-kind paths allocate destination
storage. All are invoked before `_logical_storage_for` gathers a non-compact
layout, so they can convert unrelated physical elements and then allocate a
second compact storage.

These are verified implementation paths, not a supported migration surface.
The current environment exposes only the Python backend, and the optional
pair branches were verified from their source and existing backend tests, not
by claiming that a public transfer operation already exists.

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
input 5.1.3 settles it the same way while retaining its distinct flat-layout
and public-copying rules.

`source.to(backend)` is not unqualified construction: it is the separately
named explicit transfer in 5.9, and its destination argument replaces the
active selection only for the new result it deliberately creates.

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
migrated in place. The explicit `source.to(backend)` operation in 5.9 creates
a different tensor; the source still keeps its backend and sole storage.

### T4 — Operations consume matching tensors and produce native storage

An operation runs on the **active backend** (2.1). It requires operands whose
storage belongs to that backend, and produces a result in that backend's
storage. It does not inspect its operands to decide where to run, and it does
not convert them.

### T5 — Cross-backend transfer is outside the execution contract

Combining tensors from different backends, or using a tensor while a different
backend is active, is **not** an operation this contract defines. It is not
slow, or discouraged: it is outside the contract.

The one explicit transfer contract is `source.to(backend)` (5.9). It is not
numerical execution, constructor coercion or fallback: its name and argument
state that a new independently owned tensor is requested at that destination.
No other operation inherits this exception.

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
| **`Storage`** | backend must match the active backend; see 5.1.3 ([Q4](#q4--construction-from-a-storage-object--resolved), resolved) |

The scalar, list and `array` cases are unambiguous: they are host inputs with
no backend of their own, so they take the active backend's. A `Tensor` input
already has a backend, which 5.1.1 settles. A `Storage` input also already has
one, which 5.1.3 settles without importing tensor-layout rules that a flat
storage does not possess.

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
- **Q4 is genuinely separate.** Q4 has now independently resolved construction
  from a `Storage` in
  [5.1.3](#513-construction-from-storage). An implementation of 5.1.1 must not
  collapse the two code paths: storage has no tensor layout to gather, and its
  public-copy boundary differs from private owned-storage adoption.
- **Q8 supplies the deliberate escape hatch.** R1 still forbids construction
  from moving a tensor, while [Q8](#q8--an-explicit-migration-api--resolved)
  now assigns that job exclusively to `source.to(backend)` (5.9).
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

#### 5.1.3 Construction from Storage

`Tensor(storage)` and `Tensor(storage, dtype=…)` are public construction from
an externally supplied flat buffer. They produce a **new, independently owned
tensor**. This boundary is deliberately different from the private
`_from_owned_storage` ownership-transfer path below.

**S1 — The storage backend must be the active backend.** If `storage.kind`
does not equal the active backend, construction raises the mismatch error of
T6, naming the storage backend, active backend and constructor. It does not
convert the storage, adopt its backend against the selection, or fall back to
Python storage. The check happens before a copy or dtype conversion, so a
failed construction does not read or alter the source.

**S2 — Public construction deep-copies.** On success, the result owns a new
backend-native storage object and mutable buffer. It shares neither with the
supplied `Storage`, nor with any tensor that already uses that storage. The
constructor does not take ownership away from the caller and does not mutate
the supplied object.

**S3 — Storage contributes values and dtype, not tensor layout.** When `shape`
is omitted, the result shape is `(storage.size,)`. When it is supplied, it is
validated by the existing `Shape` rules and its element count must equal
`storage.size`; otherwise construction raises. In either case strides are the
canonical contiguous strides for the result shape, offset is zero, and the
storage holds exactly those flat elements. There is no source shape, stride or
offset to preserve or gather.

**S4 — Dtype is inherited or converted on the active backend.** With no dtype
argument, the result inherits `storage.dtype`. An equal explicit dtype has the
same effect. A different explicit dtype converts the stored values and leaves
the result in native storage of the active backend. Conversion follows the
existing construction and casting rules of
[arithmetic-semantics §4.5](arithmetic-semantics.md#45-arithmetic-is-not-construction-or-casting):
out-of-range integer targets raise, float overflow yields `inf`, and
float-to-integer conversion truncates toward zero. Arithmetic wraparound does
not apply. No cross-backend conversion is involved because S1 has already
required the kinds to match.

**S5 — Mutation and versioning are independent.** The result starts at
version zero. Later mutation of the source buffer or a tensor using it is not
visible in the result; mutation of the result is not visible through the
source storage, and advances only the result's version. This follows from S2
and preserves the public ownership guarantee in
[memory-model.md](memory-model.md#ownership-and-current-limits).

**S6 — Construction is outside graph recording and gradient propagation.**
The constructor creates a plain `Tensor`, touches no graph state, creates no
node or edge and propagates no gradient relationship from a tensor that might
already use the storage. A later `Variable` wrapper has its existing semantics;
this constructor does not add any.

##### Public and internal behavioural matrix

`A` is the active backend. Rows 1–6 describe public construction; row 7 is the
private ownership-transfer boundary.

| # | Case | Outcome | Dtype and backend | Shape and layout | Ownership and mutation |
| --- | --- | --- | --- | --- | --- |
| 1 | `storage.kind == A`, dtype omitted or unchanged | succeeds | `storage.dtype`, native to `A` | default `(storage.size,)`; contiguous, offset 0 | deep copy; source and result are independent |
| 2 | `storage.kind == A`, explicit different dtype | succeeds if values are representable under §4.5 | requested dtype, converted on `A` and still native to `A` | as row 1 | new independent converted buffer |
| 3 | `storage.kind != A` | **raises** the T6 mismatch error before copying or conversion | no result; no transfer or fallback | no result | source untouched |
| 4 | Valid explicit shape with `shape.size == storage.size` | succeeds | as rows 1–2 | supplied shape; newly computed contiguous strides; offset 0 | as rows 1–2 |
| 5 | Invalid shape or `shape.size != storage.size` | raises the existing shape `TypeError`/`ValueError` or size-mismatch `ValueError` | no result | storage has no layout metadata to reinterpret | source untouched |
| 6 | Either side is mutated after successful construction | succeeds independently | unchanged | unchanged | no shared buffer; only the mutated tensor's version advances |
| 7 | `_from_owned_storage(new_storage, dtype=…, shape=…)` | adopts a valid freshly produced internal result | supplied storage's dtype and kind; active backend is not consulted | required explicit shape; contiguous strides; offset 0 | **no copy**; ownership is transferred and the producer must not retain a mutable alias |

##### The internal owned-storage boundary

`_from_owned_storage` remains sufficient; Q4 introduces no additional
abstraction and exposes no new public API. Its contract is:

- the caller supplies storage freshly allocated or otherwise exclusively
  owned for this result and relinquishes mutable use of it;
- the returned tensor adopts the exact storage object and buffer without a
  copy, owns it, uses the required explicit shape, contiguous strides, offset
  zero and an initial version of zero;
- `storage.dtype` must equal an explicitly supplied expected dtype, and
  `storage.size` must equal `shape.size`; disagreement raises rather than
  converting, reshaping values or truncating them;
- the storage's kind becomes the result tensor's backend. The helper does not
  consult the active backend and may be called while another backend is
  active. That is necessary for a kernel or other internal operation whose
  contract authorizes production on its own backend; it is adoption, not a
  transfer. Under T4 and T11, ordinary target-model dispatch must still
  produce storage on the required executing backend or raise — the helper
  does not legitimize fallback and does not repair an invalid dispatcher; and
- the helper itself does not record graph structure or gradients. Eager
  operations, graph replay, fusion and backward passes construct their result
  tensors through it, while their surrounding operation/graph layer remains
  responsible for graph semantics.

The current implementation already performs the no-copy adoption, dtype and
size checks, and metadata initialization. It cannot enforce exclusive
ownership: passing storage that another tensor or caller can mutate violates
the private precondition and creates observable aliasing. That is a caller
contract, not a reason to weaken S2 or make the private helper copy.

##### What changes from today and implementation dependencies

- **Backend mismatch:** public construction currently copies the source kind
  without reading the selection. It must instead perform S1's early check.
- **Explicit dtype:** a differing dtype currently raises in `_set_storage`.
  It must become an active-backend-native conversion under S4. Like 5.1.1 R6,
  this depends on strict cast execution: the current workload-based cast
  dispatcher may return Python storage for a small NumPy workload and cannot
  be reused unchanged.
- **Backend-native allocation:** both the unchanged-dtype copy and changed-
  dtype conversion must allocate through the matching backend without a host
  intermediate. `Storage.copy()` already supplies the first operation; the
  second needs a backend-native casting path with §4.5's numerical behaviour.
- **Internal result construction:** creation, random generation, eager forward
  and backward kernels, graph fusion and optimizer updates already transfer
  newly produced buffers through `_from_owned_storage`. Those call sites keep
  the no-copy path. Their dispatchers remain responsible for T4/T11 and may
  not use this helper to bless fallback storage.
- **Ownership:** public construction must keep copying; internal adoption must
  keep its exclusivity precondition. Shared-storage views and alias-version
  propagation remain outside this proposal.
- **Graph execution:** neither constructor path needs a graph-layer change.
  Graph-produced native storage may be adopted internally; Q5 still governs
  replay under a mismatched selection.
- **Q8:** S1 deliberately leaves public construction unable to migrate
  storage. Migration belongs exclusively to the separately named
  `source.to(backend)` contract in 5.9; Q4 does not implement it.

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

A mismatch is detected no later than the point where an operand would be
consumed. Public construction checks at its boundary (5.1.1, 5.1.3), and graph
replay checks its complete known leaf set before rebinding or execution (5.7);
ordinary eager operations validate at consumption. The error names the
tensor's backend, active backend and constructor, replay phase or operation,
so the cause is legible without a debugger. Four cases:

- **Operand against the active backend.** A NumPy tensor used while CUDA is
  active.
- **Operand against operand.** A NumPy tensor and a CUDA tensor in one
  operation, whatever the selection.
- **Operand against a graph.** A recorded computation replayed with operands
  from a different backend — see
  [Q5](#q5--graph-replay-across-backends--resolved).
- **Source against the active backend.** `Tensor(source)` where the source's
  backend is not the active one (5.1.1 R1 for a `Tensor`, 5.1.3 S1 for a
  `Storage`). Construction is not exempt from T6 merely because it produces a
  new tensor rather than consuming two.

`source.clone()` is **not** in this list. It names no backend and consumes no
second operand, so there are never two answers to disagree: it operates on the
source's backend by definition (5.1.2, C1).

### 5.4 Mutation

> **Status: implemented.** Unlike the rest of section 5, this section
> describes the package as it stands; 3.4 records the same behaviour among
> the measured ones. It does not depend on T3 or T7 being implemented,
> because it never asks where a tensor *should* live — only where it
> already does.

An in-place write updates the tensor's own storage, on its own backend. It does
**not** migrate the tensor to the host. Mutation versioning and
stale-autograd detection are unchanged (T9).

A write whose value comes from the host — `t[0] = 5.0` — transfers one scalar
to the tensor's backend. That is a host-facing write, the mirror of a
host-facing read (5.5), and is intended. A tensor supplying the values is not
a host value and is not read out to the host: it reaches the destination's
backend as native storage, and a broadcast among the values is applied there
as the mapping it is rather than by materializing repeated values on the host.

Mutation is the one boundary that answers to the tensor rather than to the
selection. Every other execution boundary runs where the active backend says,
because it is producing a new value; this one changes a value that already
exists and already has a backend. Dispatching a write on the selection would
either migrate the tensor, which T3 forbids, or refuse a write to a tensor
built under a different selection, which nothing here requires.

### 5.5 Display and host access

The public request for a Python value or display string is an explicit host
read, not an implicit transfer. The following rules are normative.

**H1 — Public host reads are permitted independently of the active backend.**
`repr`/`str`, `tolist()`, `item()`, complete integer indexing that returns a
scalar, scalar formatting and tensor equality may observe a tensor from any
resident backend under any active selection. They read the tensor's own
authoritative storage; they do not ask the active backend to execute numerical
work or require residency to match it.

**H2 — A read does not change tensor state.** It does not replace or mutate
storage: the authoritative `Storage` object and buffer retain their identity.
It does not change the tensor's backend, increment its version, change shape,
strides or offset, or create a persistent Python/host representation. There
is no host cache. Repeating the same read may repeat the transfer.

**H3 — Values are logical, not physical.** Host results enumerate the tensor's
logical elements in row-major order, respecting shape, strides and offset and
excluding unrelated physical storage. The existing flat-list behaviour of
`tolist()` remains unchanged. `item()` and complete integer indexing return
the selected logical scalar. Display and equality observe the same logical
values.

**H4 — Returned Python values are independent.** A list returned by
`tolist()` is a new container and cannot mutate or alias the tensor's storage.
A scalar or string is an ordinary Python value. Later tensor mutation cannot
retroactively change a value already returned.

**H5 — Temporaries are bounded by one public operation.** Implementations may
gather native logical values and materialize an operation-local host buffer,
list or scalar. The temporary must become unreachable when the public call
finishes and must not be stored on the tensor, in a backend cache, or in graph
state. A nested formatter should materialize once per top-level display call,
not transfer once per element.

**H6 — Native host storage avoids a transfer, not the ownership rule.** A
compact Python-backed tensor may read its authoritative buffer directly; a
non-compact Python-backed tensor may gather from it. NumPy-backed tensors may
gather with NumPy before conversion to Python values. CUDA-backed tensors
gather the requested logical region on device before the device-to-host copy,
so a view does not transfer unrelated physical values. `tolist()` still
returns an independent list in every case.

**H7 — Inspection permission is not execution permission.** `_data` must not
remain a general numerical interface. Python reference kernels may read native
Python storage only when T4 has already established Python execution with
Python-native operands. Operations, gradients, validation, broadcasting,
manipulation and graph code must not call the public host-read path to obtain
values for computation. An internal consumer either operates on native
storage through the selected backend or is rejected by T4–T6 and T11; it may
not label its transfer “inspection.”

**H8 — Scalar indexing separates inspection from tensor-producing slicing.**
Complete integer indexing resolves the coordinate from tensor metadata,
gathers from the tensor's own resident storage and returns a Python scalar;
like `item()`, it is permitted under any active selection. Indexing that
returns a `Tensor` is an ordinary backend-native layout operation governed by
T4 and T10. The scalar permission cannot make a mismatched tensor-producing
slice kernel or Python fallback valid.

| Case | Required host-access behaviour | Persistent effect |
| --- | --- | --- |
| Python-backed tensor | Read the authoritative buffer directly; gather locally when the layout is non-compact. `tolist()` still creates an independent list. | None |
| NumPy-backed tensor | Gather logical values with NumPy, then create the requested Python scalar, flat list or string. | None |
| CUDA-backed tensor | Gather the logical selection on device, transfer only that selection to an operation-local host buffer, then create the Python result. | None |
| Non-contiguous or offset tensor | Follow shape, strides and offset in logical row-major order; never expose or transfer unrelated physical elements merely to fit a host helper. | Layout and authoritative storage remain unchanged. |
| Repeated reads | Perform a fresh observation; NumPy conversion or CUDA transfer may happen again. | No representation from the earlier read is reused from the tensor. |
| Mutation between reads | The later read observes current authoritative values and version; it cannot consult stale host state. | The reads themselves do not advance the version. |
| Different active backend | Read the tensor's own resident storage using H1/H8; do not switch the active backend or convert the tensor to the active selection. | Storage identity, residency and selection remain unchanged. |

This resolves [Q7](#q7--repeated-host-reads--resolved) by accepting visible,
repeatable transfer cost rather than restoring a second representation.

### 5.6 Serialization

There is nothing to preserve (3.10), so no behaviour is specified. If
serialization is added, whether a stored tensor records its backend is
[question Q6](#q6--serialization).

### 5.7 Graph execution

A recorded computation describes mathematical operations, graph identities
and data flow. It does **not** acquire a backend merely because one was active
while the structure was recorded. Compatibility is decided from the tensors a
particular pass will read and the backend active for that pass.

**G1 — Recording time does not bind execution.** A backend-neutral graph may
execute under a different active backend from the one used while recording.
Recording-time selection is neither stored as graph semantics nor compared at
replay solely for having been the recording selection. This avoids a redundant
graph-level selection system beside `get_backend()`.

**G2 — Every tensor read by a pass must be native to the active backend.** For
a forward replay this set includes new replay inputs plus every bound leaf the
program retains: parameters, captured Variables, Tensor constants and scalar
tensor leaves. For a reverse pass it includes the explicit or generated seed,
the forward inputs and saved intermediate/output values each VJP reads, and
gradient terms as they are produced. A captured constant is an operand, not an
exception.

**G3 — Known leaf mismatches are rejected before replay mutates bindings or
executes an instruction.** Replay validates all incoming Tensor arguments and
all retained bound leaves against the active backend as one preflight. A
mismatch raises the T6 error naming the resident backend, active backend and
the input/capture or replay phase. It does not rebind input Variables, execute
a prefix of the program, convert anything, or update output Variables.
Mixed-backend replay inputs necessarily fail this check: at most one kind can
equal the active backend.

This guarantee is deliberately narrower than transactionality. Once preflight
succeeds, instructions execute in order and result Variables are updated as
they are produced. A later unsupported operation, internal invariant failure
or wrong-kind result may therefore leave already-produced intermediate or
output Variables updated; replay does not roll them back. Input and captured
tensors are not mutated by ordinary operations. The implementation must not
describe the whole replay as atomic.

**G4 — Prior-pass results do not pin a new forward pass.** A bound output or
intermediate Tensor from an earlier replay is overwritten before its slot is
read in the new pass, so its old backend alone is not an incompatible capture.
Any cached value that *is* read as an operand is part of G2 and must match.
This distinction permits one backend-neutral program to replay with entirely
new, compatible inputs on another backend without treating stale outputs as
captures.

**G5 — Dispatch retains its ordinary responsibilities.** After graph-level
preflight, each instruction calls the same operation entry point as eager
execution. Operand validation enforces T4–T6 at consumption, dispatch selects
the active backend's kernel, and `_from_owned_storage` may adopt only the
native result the valid dispatcher produced. If the selected backend cannot
execute an operation conformingly, dispatch raises
`BackendOperationUnsupportedError` naming the operation, dtype and backend.
Replay must not catch that error to invoke another backend.

Eager `Variable` execution reaches the same rule from the other direction: it
records one operation node, compiles the new fragment and immediately calls
the operation. It does not perform a whole-graph replay preflight. A mismatch
is rejected by ordinary operand validation before that operation's kernel runs
or result binds, but the already-recorded structural fragment may remain until
collected. This is harmless graph metadata, not a partially computed tensor,
and is not a reason to transfer the operands.

**G6 — Execution artifacts are replaceable; tensors are not transferable.**
The canonical instruction sequence and current fusion-range metadata are
backend-neutral. A backend-specific compiled or fused artifact, now or in a
future implementation, is valid only for the backend and capability key it
was created for. Replay may select, compile, regenerate or cache a compatible
artifact, because that changes executable code rather than tensor residency.
It may also decline a fused plan and execute the same instructions ordinarily
on the same backend. It must never use an artifact mismatch to convert data or
fall back to another backend. If neither fused nor ordinary execution is
supported, G5's unsupported-operation error applies.

**G7 — Cache compatibility is not graph compatibility.** The current compiled
trace signature includes the active backend and retraces on a change. Under
this contract the recording backend is not a semantic guard. A backend may
participate in an execution-artifact cache key, but a miss means select or
regenerate an execution plan; it does not by itself make the mathematical
graph incompatible. Shape, dtype, static-argument and structural guards remain
independent validity requirements.

##### Replay compatibility matrix

`A` is the backend active for the pass. A "retained leaf" means a parameter,
captured Variable, Tensor/scalar constant or any other bound value the pass
will read.

| Scenario | Required result | Detection and state on failure |
| --- | --- | --- |
| Recording and replay both use `A`; all read tensors are native to `A` | replay succeeds; every instruction and result remains on `A` | ordinary operand validation and dispatch still apply |
| Replay changes to backend `B` but retains an `A`-backed captured tensor | **raises** a T6 mismatch | replay preflight identifies the captured leaf before input rebinding or instruction execution; tensors and result bindings are unchanged |
| Graph was recorded under `A`, then replayed under `B` with new `B` inputs, no incompatible retained leaf/artifact, and all non-backend guards satisfied | replay is permitted; recording backend is irrelevant | inputs are rebound only after successful preflight; results and intermediates are native to `B` |
| Replay inputs have different storage kinds, or an input kind differs from `A` | **raises** a T6 mismatch | replay preflight names the offending input and active backend; no input binding changes |
| A captured constant or parameter is not native to `A` | **raises** exactly as for any tensor operand | replay preflight; capture status grants no conversion privilege |
| A cached compiled/fused artifact is for another backend | graph remains compatible if a valid `A` plan exists | select/regenerate an `A` artifact or execute ordinary instructions on `A`; artifact-cache state may change, tensor data may not be transferred |
| Backend `A` lacks a required operation | **raises** `BackendOperationUnsupportedError` | raised by ordinary dispatch at that instruction; earlier results may already have been updated because replay is not transactional |
| Forward ran on `A`, backward is requested under `B` while saved values remain on `A` | **raises** a T6 mismatch | reverse preflight occurs before seed computation, VJP execution or `.grad` publication; rerun forward with compatible `B` inputs/captures before differentiating on `B` |

Rows two and three are the core Q5 distinction: changing selection is neither
automatically invalid nor automatically valid. Tensor residency, retained
state and backend capability decide.

##### What changes from today and implementation dependencies

- **Replay preflight is new.** Structural and compiled replay currently rebind
  public inputs before `Computation.forward()`, and direct computation replay
  performs no all-leaf backend check. The check must cover new inputs and
  `_leaf_variables()` before either action. Ordinary per-operation validation
  remains defence in depth and covers dynamically produced values.
- **The compiled trace guard is too broad.** `_call_signature()` stores
  `get_backend()` with shape, dtype and static arguments, forcing a retrace on
  every backend change. Backend-neutral instructions can be reused; any truly
  backend-specific artifact needs its own compatibility key rather than making
  recording selection graph semantics.
- **Captured values are already discoverable.** Compiler leaf slots include
  public boundaries, model parameters and Tensor/scalar leaves, so no new
  public capture registry or graph API is required. The implementation must
  distinguish public input bindings from retained captures only to produce a
  useful error and to avoid mutating bindings before validation.
- **Scalar graph constants must become backend-native.** Eager Variable
  arithmetic currently materializes a scalar leaf through `_from_values`,
  which always creates `PythonStorage`. Under a non-Python active backend that
  would make the graph incompatible at birth once implicit conversion is
  removed. Scalar normalization must allocate on the active backend while
  preserving the existing promotion and conversion rules; this is a T1/T4
  dependency, not permission to special-case constants during replay.
- **Backend-native result construction remains internal.** Eager, replayed,
  fused and backward kernels continue to transfer fresh storage through
  `_from_owned_storage`. As 5.1.3 requires, that helper does not validate the
  active selection and therefore cannot legitimize a dispatcher that returned
  the wrong kind.
- **Fusion needs no new graph format.** The existing range/step plan is
  backend-neutral. CUDA's typed `RawKernel` caches remain backend-owned and may
  compile on demand. A future graph-owned executable must be invalidated or
  regenerated on incompatibility; retaining one and moving data to suit it is
  forbidden.
- **Strict dispatch remains a dependency.** Many non-arithmetic forward and
  VJP dispatchers still use workload-based Python fallback. A graph cannot
  satisfy G5 while those paths can return storage from another backend. This
  is the same T4/T11 implementation gap, not a graph-specific exception.

### 5.8 Differentiation

A gradient is produced on the backend of the tensors it is computed from, and
the VJP kernels are subject to T4 and T6 like any other operation. The
numerical contract — the region tables, accuracy bounds and classification
rules of [Arithmetic semantics §12.7](arithmetic-semantics.md#127-differentiation-d7)
— is untouched (T9).

The active backend for backward need not equal the historical recording
backend, but it must equal the residency of every saved forward value and seed
that reverse execution will read. Forward-under-one-backend,
backward-under-another therefore raises while the saved pass remains on the
first backend. That is the intended consequence of T5: it works today only
through the implicit conversion that this proposal removes. A fresh successful
forward replay under the second backend, with compatible inputs and captures,
refreshes saved states and may then be differentiated there.

Reverse compatibility is checked after the existing mutation/version
validation and before a default seed is allocated or an explicit seed is
cast. A supplied seed is an operand and must already be native to the active
backend; seed casting may change dtype on that backend but may not migrate it.
Saved operand, output and intermediate tensors follow G2 even when a VJP needs
only a subset of them.

`backward()` retains its existing publication guarantee: no `.grad` field is
cleared or replaced until the complete reverse pass succeeds. A backend
mismatch or unsupported VJP therefore leaves published gradients unchanged,
although local gradient buffers and, for `create_graph=True`, unreachable
partial structural records may have been created before a later non-preflight
failure. `grad()` remains functional and never publishes `.grad`.

Higher-order differentiation adds no exception. A recorded VJP holds
operations as backend-neutral graph structure, while the Variables and Tensors
it reads and produces obey the same active-backend checks. Gradient
accumulation, shape reduction, dtype restoration, saved-state version checks
and numerical semantics remain unchanged; only an implicit backend transfer is
removed.

### 5.9 Explicit backend migration

`source.to(backend)` is the sole public request to copy a tensor to another
backend. It is the explicit exception to T5, not an exception to T3: the
source is never changed, and the result is a distinct tensor with one
authoritative destination-native storage. Constructors, ordinary operations,
graph replay and fallback do not acquire migration permission from this API.

**M1 — The destination is explicit and concrete.** `backend` names `python`,
`numpy` or `cuda`, is normalized using the existing backend-name conventions,
and is checked for availability without changing the active backend or process
default. `auto` is a selection policy rather than a destination and is not
accepted by this method. The requested destination need not match the active
backend or process default.

**M2 — The source is immutable.** Migration does not mutate or replace the
source's storage, change its backend, shape, strides or offset, advance its
mutation version, or populate a cache. Its authoritative `Storage` object and
buffer retain their identity even if the transfer fails.

**M3 — Ownership is independent.** The result owns a new `Storage` object and
new buffer. This also applies when source and destination backends are equal:
`.to(source_backend)` returns a copy, never `self` and never a shared view. The
result starts with mutation version zero, and later writes to either tensor
cannot affect the other.

**M4 — Logical contents are preserved.** Shape, dtype, numerical values and
logical row-major element order are unchanged. A non-contiguous or offset
source is gathered on its resident backend before transfer, so unrelated
physical elements are not copied. The result is compact and contiguous, with
offset zero, contiguous strides and storage size equal to `shape.size`.

**M5 — Migration does not cast.** `.to()` has no dtype parameter and performs
no numerical dtype conversion. The destination must represent the source's
existing `DataType` exactly or the call fails.

**M6 — No representation is cached.** Source-native gathers and host staging
buffers may exist for the duration of the call. They are not retained on the
source or result, placed in graph state, or reusable by a later migration.
Only the result's destination-native storage survives.

**M7 — Failure is source-atomic.** A non-string destination raises
`TypeError`; an unknown name or `auto` raises `ValueError` using the existing
backend-name style; and a recognized but unavailable optional backend raises
`BackendUnavailableError`. A destination unable to represent the preserved
dtype raises `TypeError` before transfer where that can be determined.
Allocation and provider transfer failures propagate their native exception
with source and destination context. No failure returns a partial tensor or
changes any source state; partially allocated temporaries are discarded.

#### 5.9.1 Backend-pair behaviour

No row is a public capability today because `.to()` does not exist (3.11).
The “existing primitive” column records source-verified building blocks; the
optional NumPy and CUDA branches were not runnable in the current
Python-only environment.

| Source → destination | Target one-shot path | Existing primitive and required change |
| --- | --- | --- |
| Same backend | Gather logical values natively if needed, then `Storage.copy()` or an equivalent independent native copy | `copy()` already supplies independent storage; `convert_storage` cannot be used because its same-kind branch returns the source object |
| Python → NumPy | Gather into the logical Python buffer, then allocate an independent NumPy array | `numpy.frombuffer` exists but aliases the Python buffer; the destination must copy |
| Python → CUDA | Gather in Python, then perform one host-to-device copy; an operation-local NumPy view may stage it | The current converter stages through `numpy.frombuffer` and `cupy.asarray`; detach this path from caching and retain only the CUDA allocation |
| NumPy → Python | Gather with NumPy, then copy the logical bytes/values into `array.array` | The bytes path already creates independent Python storage; it must receive only the logical gather and must not cache it |
| NumPy → CUDA | Gather with NumPy, then use `cupy.asarray` for one host-to-device copy | The pair primitive exists; give its new CUDA storage exclusive ownership and no cache entry |
| CUDA → Python | Gather logical values on device, use `cupy.asnumpy` once, then copy into `array.array` | Host staging is necessary; the current path transfers the physical buffer too early and must be reordered |
| CUDA → NumPy | Gather logical values on device, then use `cupy.asnumpy` for one device-to-host copy | The pair primitive already returns an independent NumPy allocation; remove cache coupling and transfer only logical values |

The package's seven declared dtypes have names understood by the current
NumPy and CuPy storage constructors, but M5 does not infer universal provider
support from that fact. Availability and exact representation are validated
for the requested pair at call time.

#### 5.9.2 Graph and differentiation boundary

`.to()` returns a plain independent `Tensor`. It does not accept or return a
`Variable`, create an operation node or gradient edge, migrate a parameter or
published gradient, rewrite a captured constant, copy saved forward values,
or move an entire graph. Existing Variables, instructions, captures and saved
states continue to refer to their original tensors.

Calling `variable.data.to(backend)` therefore only produces an untracked
tensor. Assigning that result back to `variable.data` is a separate explicit
rebinding under the existing data-generation and stale-state rules; `.to()`
itself performs no rebinding. A migrated tensor may be supplied as a new graph
input, but Q5 still requires every input and retained capture to match the
active backend for that replay. Forward and backward compatibility rules in
5.7–5.8 are unchanged, and no `Variable.to()` or automatic graph migration is
introduced.

The public copying operations consequently remain distinct:

| Operation | Backend selection | Ownership |
| --- | --- | --- |
| `Tensor(source)` | Active backend must match source | Independent copy |
| `source.clone()` | Source backend | Independent copy |
| `source.to(backend)` | Explicit concrete destination | Independent copy |

`_from_owned_storage` remains the private no-copy adoption boundary for a
fresh exclusive result. It is not a migration API, while `.to()` must first
create the independently owned destination storage that it may then adopt.

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
| `tensors/tensor.py` — `_mutable_data` | Converted to host and installed it | **Done.** Gone. Mutation acts on the tensor's own backend (5.4), through `execute_assign_indices`, which dispatches on the destination storage's kind. |
| `tensors/tensor.py` — `_data` | Converts through `_storage_for("python")`, retains a full physical host representation, then gathers logical values; also serves internal numerical code | **Restricted or replaced.** Public inspection gets an uncached logical host-read path governed by H1–H6. Internal numerical consumers do not get access to it (H7). |
| `tensors/tensor.py` — `tolist`, `item`, display/formatting, equality and scalar `__getitem__` | Share `_data`; display can request it repeatedly, equality materializes two lists, and scalar indexing dispatches a slice before `item()` | **Changed.** Each top-level observation follows H1–H8, materializes at most operation-local host state and returns independent Python values. Elementwise comparison remains ordinary backend execution. |
| `tensors/tensor.py` — `__init__` | Host storage for scalars/lists; copies a Tensor's or Storage's kind; a Tensor dtype change routes through host values, while a Storage dtype change raises | **Changed.** T1 for host inputs; 5.1.1 for a `Tensor` input; 5.1.3 for a `Storage` input, including the early backend check, public deep copy and backend-native dtype conversion. |
| `tensors/tensor.py` — `_from_owned_storage` | Adopts the exact supplied internal storage without checking the active backend; validates dtype and size | **Contract retained and made explicit.** 5.1.3 reserves it for exclusive, newly produced internal results. It remains no-copy and selection-independent; dispatch remains responsible for T4/T11. |
| `tensors/tensor.py` — `clone` | Defined as `Tensor(self)`; already returns the source's backend whatever the active backend is | **Behaviour unchanged; the definition must change.** 5.1.2 requires exactly what `clone()` already does, but `Tensor(self)` stops expressing it once the constructor gains R1, because it would then raise whenever the source's backend and the active backend differ. |
| `tensors/tensor.py` — `to` | Does not exist | **Added.** Implements M1–M7 as the sole public migration operation; returns a compact, version-zero, independently owned plain Tensor and never changes the source or selection. |
| `tensors/backend/dispatch/manipulation/cast.py` — `execute_cast` | Falls back to the Python reference below a workload threshold | **Changed.** 5.1.1 R6 requires a dtype conversion to produce the active backend's storage, which T4 already requires of any operation. See the dependency note in 5.1.1. |
| `tensors/backend/conversion.py` — `convert_storage` | The only cross-backend conversion; serves `_storage_cache`, returns the source for same-kind requests and can alias Python storage from NumPy | **Cache-oriented interface gone.** Its pair primitives may be refactored behind `Tensor.to` only, with logical gather before transfer, mandatory independent ownership and no cache (5.9.1). |
| Python reference kernels and `_data` consumers in `tensors/operations`, `tensors/graph` and `tensors/utils` | Read `_data`, so NumPy/CUDA operands can be materialized and retained on host during internal computation | **Changed.** Python kernels read Python-native operands only after dispatch enforces T4. Other internal consumers use backend-native operations or dispatch; none may use the public host-read permission as fallback (H7). |
| `tensors/backend/numpy/conversion.py` — `tensor_to_logical_array`, `_operand` | Each begins by asking for a NumPy representation. A third, `_arithmetic_operand`, has since been deleted: arithmetic now lowers its operands inline in each dispatcher branch | **Simplified.** Under T4 the operand already is one; what remains is dtype handling and the logical gather. |
| `tensors/backend/cuda/conversion.py` — the same two, plus `_widen`, `_narrow`, `_working_values` | As above, plus the PTX-based binary32 widening | As above. The subnormal-preserving widening is numerical (T9) and stays exactly as it is. |
| `tensors/backend/config.py` — `set_backend` | Reassignable at any time (3.9) | **Changed.** Sets the process default and locks it on first tensor storage; same-backend calls are accepted, a different backend raises (T7). |
| `tensors/backend/config.py` — `use_backend` | Scoped context-local override, restored in a `finally` | **Unchanged**, and exempt from the lock. T8 makes the existing restore-on-exception behaviour a requirement rather than an implementation detail. |
| `tensors/backend/config.py` — `get_backend` | Override if in scope, else the process default | **Unchanged.** It already answers the active-backend question (2.1). |
| `tensors/backend/config.py` — backend-name validation | `_resolve_backend` validates selection names, resolves `auto`, checks optional availability and is private to selection | **Shared internally without selecting.** Migration reuses normalization, errors and availability checks but rejects `auto` as a non-concrete destination (M1, M7); it never reads or writes selection state. |
| `tensors/backend/{python,numpy,cuda}/storage.py` — `copy` and constructors | Provide native contiguous allocation and independent same-kind copy; constructors can retain an input buffer unless copying is requested | **Reused carefully.** Same-kind migration copies; cross-kind construction receives an exclusively owned buffer or is forced to copy, especially for Python → NumPy. Unsupported representations fail under M7. |
| `tensors/backend/policy.py` | Workload thresholds choose a path | Untouched by this proposal; still governed by [backends.md](backends.md#execution-requirements). |
| Creation ops and `tensors/init/` | Mixed agreement with the selection (3.1) | **Changed.** T1 makes them uniform. |
| `tensors/graph/graph.py` — structural and compiled replay | Rebinds public Tensor inputs before execution; compiled signatures include `get_backend()` and retrace when it changes | **Changed.** Preflight all incoming and retained leaf tensors before rebinding. Recording backend is removed as a semantic trace guard; backend-specific artifact compatibility remains separate (5.7 G1–G7). |
| `tensors/graph/computation/computation.py` — forward and backward | Seeds leaves and dispatches instructions without a whole-pass residency check; reverse validation covers mutation state only | **Changed.** Forward validates bound leaves; reverse validates saved values and the seed before VJPs. Instruction order, local buffers, result binding, demand analysis and delayed `.grad` publication stay unchanged. |
| `tensors/graph/computation/fusion.py` and backend fusion caches | Graph-owned fusion ranges/steps are backend-neutral; CUDA owns typed compiled-kernel caches | **Contract retained.** Same-backend plan fallback remains permitted. Backend artifacts may be selected or regenerated, never used to justify tensor transfer or cross-backend fallback (G6). |

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

Q1 through Q5, Q7 and Q8 are **resolved**. Each decision is normative
elsewhere — T7 and T8 for the lifecycle, 5.1.1 for construction from a
`Tensor`, 5.1.3 for public and internal construction from `Storage`, 5.7–5.8
for graph replay and differentiation, 5.5 for host access, and 5.9 for
explicit migration. The entries here remain so rejected alternatives stay on
record. Only Q6 remains **open** and deferred.

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
| **A. Raise — ADOPTED** | Consistent with T6: construction is not a transfer mechanism. Deliberate movement uses the separately named `source.to(backend)` operation. |
| **B. Copy to the active backend** | Convenient, and a natural reading of "construct a tensor here from these values". But it is a cross-backend transfer under an ordinary-looking constructor, which is the pattern T5 removes. |
| **C. Copy the source's backend** | Today's behaviour (3.5). Construction then ignores the active backend, contradicting T1. |

**Adopted: A.** B hides the same transfer behind a different spelling; C
contradicts T1. Under A an ordinary constructor acquires the active backend;
the explicit migration contract in 5.9 acquires its named destination for a
new result. No expression quietly moves data across a bus.

Same-backend construction was specified at the same time, because resolving
only the cross-backend half would have left the more common case undefined.
5.1.1 states both, and its matrix row 3 is this decision.

**Cloning is not covered by A.** `source.clone()` asks a different question and
gets a different answer: it produces a tensor on the **source's** backend
whatever the active backend is, and never raises a mismatch. That is 5.1.2, and
it is the reason `clone()` cannot remain defined as `Tensor(self)`. A applies
to the constructor, not to every operation that copies a tensor.
**Relationship to Q8.** A leaves the constructor unable to move a tensor
between backends, which gives Q8 its force. Q8 now assigns migration to the
explicit, separately named `source.to(backend)` operation in 5.9, never to
this constructor.

### Q4 — Construction from a Storage object — RESOLVED

*Resolved in [5.1.3](#513-construction-from-storage), which is normative. The
entry is kept so the rejected alternatives stay on record.*

`Tensor(storage)` where the storage belongs to a different backend. The same
three options as Q3 apply, with one difference: a `Storage` is an internal
type, so the public surface is narrower and the compatibility cost of raising
is lower.

| Option | Consequence |
| --- | --- |
| **A. Raise — ADOPTED** | Consistent with Q3 and T6: public construction is not a transfer mechanism. Matching-backend construction deep-copies; internal owned-storage construction remains a separate no-copy boundary. |
| **B. Copy to the active backend** | Hides a cross-backend migration inside an ordinary constructor and contradicts T5. |
| **C. Copy the storage's backend** | Today's behaviour. It makes construction ignore the active selection and contradicts T1. |

**Adopted: A.** The public constructor checks `storage.kind` against the active
backend before copying or casting, and raises on disagreement. On agreement it
creates independent ownership, inherits or converts dtype on that backend,
and establishes a one-dimensional default shape or a validated explicit shape
with contiguous strides and offset zero.

The separation from Q3 remains important. A `Storage` has no shape, strides or
offset, so tensor gather and layout-preservation rules have no counterpart.
The separation from `_from_owned_storage` is equally important: that private
path adopts freshly produced, exclusively owned native storage without copying
and without applying the public constructor's active-backend check. It is an
internal ownership transfer, not public construction or migration.

### Q5 — Graph replay across backends — RESOLVED

*Resolved in [5.7](#57-graph-execution) and
[5.8](#58-differentiation), which are normative. The entry is kept so the
rejected alternatives stay on record.*

A computation recorded under one backend, replayed under another.

| Option | Consequence |
| --- | --- |
| **A. Execution-time residency — ADOPTED** | Graph structure carries no recording backend. Replay is valid when every tensor the pass reads is native to the active backend and required kernels/artifacts are available; otherwise it raises. A compatible graph may execute under a different backend from recording. |
| **B. Recording backend is permanent** | Simple validation, but rejects a backend-neutral graph with wholly new compatible inputs and duplicates the active-backend mechanism in graph state. The current graph and instruction representations do not require it. |
| **C. Replay converts or rebuilds values for the active backend** | Preserves today's cross-backend convenience by restoring the implicit transfer T5 removes. It also obscures whether captures, seeds and saved forward values were moved. |

**Adopted: A.** The graph is mathematical structure; bound and supplied
tensors carry residency. This is stricter than today's execution because no
operand, capture, intermediate, saved value or gradient is converted, but less
restrictive than binding the graph forever to recording time. A graph recorded
under NumPy may replay under Python with wholly Python-native inputs and no
incompatible retained leaves. The same replay raises before execution if it
still captures a NumPy tensor.

Forward on one backend and backward on another is not directly valid, because
the saved forward values belong to the first backend. It becomes valid only
after a fresh forward replay on the second backend with compatible values.
Backend-specific executable artifacts may be regenerated or replaced because
code is not tensor data; this does not authorize migration or fallback.

### Q6 — Serialization

No serialization exists (3.10), so this is a question about future work.

| Option | Consequence |
| --- | --- |
| **A. Serialize values and dtype only** | A loaded tensor takes the active backend, like any host input (T1). Portable across machines with different hardware. |
| **B. Serialize the backend too** | Round-trips exactly. A file written on a CUDA machine cannot be loaded on one without a device, unless loading falls back — which T11 forbids. |

**Recommended: A** if serialization is added. Nothing needs deciding until it
is.

### Q7 — Repeated host reads — RESOLVED

*Resolved in [5.5](#55-display-and-host-access), which is normative. The entry
is kept so the rejected alternative stays on record.*

T3 removes the cache, so `tolist()` twice transfers twice (5.5).

| Option | Consequence |
| --- | --- |
| **A. Accept the cost — ADOPTED** | Simple and honest: each host read may transfer, and the caller can see how many they asked for. Printing a large device tensor in a loop is slow. Operation-local buffers are permitted but no representation survives the call. |
| **B. Cache host values** | Restores a second representation, which is what T3 removes, and raises a staleness question on mutation. |

**Adopted: A.** B is the cache under another name. A host read is explicit,
leaves the authoritative storage and tensor backend unchanged, and may perform
the same transfer again when repeated. The permission is limited to returning
a Python value or display representation; internal kernels and graph work
remain subject to T4–T6 and T11.

### Q8 — An explicit migration API — RESOLVED

*Resolved in [5.9](#59-explicit-backend-migration), which is normative. The
entry is kept so the rejected alternative stays on record.*

With implicit transfer gone, an ordinary operation or constructor does not
move a tensor between backends.

| Option | Consequence |
| --- | --- |
| **A. None** | The strictest model. A program uses one backend, or constructs each tensor under the selection it belongs to. Q3 option A has no direct migration path. |
| **B. `Tensor.to(backend)` — ADOPTED** | Transfers are visible and greppable. A new independent Tensor is created at the concrete destination without changing the source, selection, graphs or gradients. |

**Adopted: B.** `source.to(backend)` is the only public migration request. It
always returns an independently owned compact Tensor, even for a same-backend
destination, and preserves dtype and logical values. It does not excuse a
backend mismatch in construction, numerical execution, graph replay or
differentiation, and it does not restore alternative-representation caching.

## Related documents

- [Numerical backends](backends.md) — selection, execution requirements and
  residency **as implemented**.
- [Tensor memory model](memory-model.md) — layout, ownership and the deferred
  view work.
- [Arithmetic semantics](arithmetic-semantics.md) — the numerical contract,
  unaffected by this proposal.
- [Package structure](package-structure.md) — where the affected modules live.
