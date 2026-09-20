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
> Section 8 records the questions this document does **not** settle. They are
> design decisions, not implementation details, and they are listed with their
> alternatives and consequences rather than resolved in passing.

## 2. The problem

A `Tensor` today can hold several storage representations of the same values —
one authoritative, and others produced on demand and kept in a cache. The cache
exists so that a tensor built on one backend can be used by an operation on
another without the caller noticing.

That convenience is the source of the difficulty:

- **A tensor's backend is not a property of the tensor.** It is whichever
  representation an operation last needed, which makes residency a question
  with no stable answer.
- **Transfers are invisible.** A host-to-device copy costs orders of magnitude
  more than the arithmetic around it, and nothing in the calling code shows
  where one happened.
- **Mismatches are hidden rather than reported.** Combining a host tensor with
  a device tensor silently converts one of them, so a program that has
  accidentally left half its data on the wrong side of the bus still runs, only
  slowly.
- **Every kernel pays for the ambiguity.** Each one begins by asking for its
  operands in its own representation, because it cannot assume what it was
  given.

The proposal removes the ambiguity rather than managing it: a tensor has one
storage representation, it is the selected backend's, and a mismatch is an
error.

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

The selection has no say; the source's representation is copied — but only
while the dtype agrees. `Tensor(src, dtype=...)` with a *different* dtype takes
a separate path that reads the values on the host and builds `PythonStorage`,
so a dtype-changing construction moves the tensor to the host whatever the
source and the selection were. Two inputs that differ only in dtype therefore
produce tensors on different backends.

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

### 3.9 There is no serialization

No `save` or `load` exists in the public API, and `Tensor` is not picklable
(`pickle.dumps` succeeds, `loads` raises `TypeError: Invalid shape`). There is
therefore no serialization behaviour to preserve — see
[question Q6](#q6-serialization).

## 4. The target contract

### T1 — Selection determines construction and execution

The active backend decides both where a new tensor's values live and where an
operation runs. `Tensor([1.0, 2.0])` under NumPy selection produces NumPy
storage; under CUDA, device storage. Creation operations follow the same rule,
so `zeros`, `full`, `eye`, `arange`, `linspace`, the random samplers and the
initializers all agree with the selection. The inconsistency in 3.1 disappears
because there is one rule.

### T2 — Each backend uses its own native storage

Python uses `array.array`, NumPy uses `numpy.ndarray`, CUDA uses device-resident
`cupy.ndarray`. This is unchanged and remains an internal detail, not a second
public array API.

### T3 — One authoritative representation

A tensor has exactly one storage. **No alternative representation is retained
to support an implicit transfer.** The storage's `kind` is the tensor's
backend, it is fixed for the tensor's lifetime, and it is answerable at any
time without qualification.

### T4 — Operations consume matching tensors and produce native storage

An operation under a given selection requires operands whose storage belongs to
that backend, and produces a result in that backend's storage. It does not
inspect its operands to decide where to run, and it does not convert them.

### T5 — Cross-backend transfer is outside the execution contract

Combining tensors from different backends, or using a tensor under a selection
other than its own, is **not** an operation this contract defines. It is not
slow, or discouraged: it is outside the contract.

### T6 — A mismatch is an error

A mismatch raises, naming the tensor's backend, the active selection and the
operation. It never converts, never falls back, and never succeeds quietly.
The error is the mechanism by which T5 is enforced rather than merely advised.

### T7 — `set_backend` is a one-time selection

`set_backend` configures the backend for normal execution, and the model is
one selection per process, made before tensors exist.

- **Before initialization**, any available backend may be selected.
- **Initialization** is the first construction of a tensor's storage under the
  current selection — the point from which tensors exist that the selection
  describes. It is a property of the process, not of any one tensor.
- **After initialization**, selecting the *same* backend again is accepted and
  has no effect. It is idempotent by design, so library code may assert its
  requirement without needing to know whether it was first.
- **After initialization**, selecting a *different* backend raises. Allowing it
  would leave existing tensors describing a backend that is no longer selected,
  which T3 and T6 together make unusable — a failure at the next operation
  instead of at the call that caused it.

[Question Q1](#q1-the-scope-of-the-one-time-rule) records what "initialization"
should be measured by, which is not settled here.

### T8 — `use_backend` is scoped and moves nothing

`use_backend` selects a backend for the duration of a block and restores the
previous selection on exit, nesting as it does today (3.8). It **does not**
mutate, migrate, convert or copy any existing tensor. Tensors created inside
the scope belong to the scoped backend and outlive it; tensors created outside
it are unaffected and, under T6, are not usable inside it.

The relationship between `use_backend` and T7's one-time rule is
[question Q2](#q2-use_backend-against-the-one-time-rule).

### T9 — Tensor properties are preserved

dtype, shape, strides, offset, ownership, mutation versioning and every rule in
[Arithmetic semantics](arithmetic-semantics.md) are unchanged. This proposal
concerns where values live, not what they are or what operations compute from
them.

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

### 5.1 Construction

| input | required behaviour |
| --- | --- |
| Python scalar | storage for the active backend, shape `()` |
| nested list | flattened and stored on the active backend |
| `array.array` | values read on the host, stored on the active backend |
| **`Tensor`** | see [Q3](#q3-construction-from-a-tensor-of-another-backend) |
| **`Storage`** | see [Q4](#q4-construction-from-a-storage-object) |

The scalar, list and `array` cases are unambiguous: they are host inputs with
no backend of their own, so they take the selection's. Construction from a
`Tensor` or a `Storage` is not, because the input already has a backend, and
both are recorded as open questions rather than decided here.

### 5.2 Nested backend contexts

Nesting is unchanged (3.8, T8). A tensor's backend is the selection in force
when it was constructed, and it does not change when a scope exits. A program
that constructs under `use_backend("cuda")` and then reads `tolist()` outside
the scope still reads a device tensor's values, because host-facing access
(5.5) does not depend on the selection.

### 5.3 Backend mismatch

A mismatch is detected where the operand is consumed and raises there. The
error names the tensor's backend, the active selection and the operation, so
the cause is legible without a debugger. Three cases:

- **Operand against selection.** A NumPy tensor used under CUDA selection.
- **Operand against operand.** A NumPy tensor and a CUDA tensor in one
  operation, whatever the selection.
- **Operand against a graph.** A recorded computation replayed with operands
  from a different backend — see [Q5](#q5-graph-replay-across-backends).

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
transfers again. [Question Q7](#q7-repeated-host-reads) records whether that
cost is acceptable.

### 5.6 Serialization

There is nothing to preserve (3.9), so no behaviour is specified. If
serialization is added, whether a stored tensor records its backend is
[question Q6](#q6-serialization).

### 5.7 Graph execution

A recorded computation describes operations and operands, not a backend. Under
the target model its inputs have backends, so replay is constrained by them
rather than by where the graph was built. Replay under a selection matching
its inputs produces results on that backend, as it does today (3.7); replay
against a different one is a mismatch under 5.3, which is a behaviour change
from 3.7 and is [question Q5](#q5-graph-replay-across-backends).

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
| `tensors/tensor.py` — `__init__` | Host storage for scalars/lists; copies a Tensor's or Storage's kind | **Changed.** T1 for host inputs; Q3 and Q4 for the rest. |
| `tensors/backend/conversion.py` — `convert_storage` | The only cross-backend conversion | **Gone**, unless a deliberate migration API is added — see [Q8](#q8-an-explicit-migration-api). Its one caller is `_storage_for`. |
| `tensors/backend/numpy/conversion.py` — `_view`, `_operand`, `_arithmetic_operand` | Each begins by asking for a NumPy representation | **Simplified.** Under T4 the operand already is one; what remains is dtype handling and the logical gather. |
| `tensors/backend/cuda/conversion.py` — the same three, plus `_widen`, `_narrow`, `_working_values` | As above, plus the PTX-based binary32 widening | As above. The subnormal-preserving widening is numerical (T9) and stays exactly as it is. |
| `tensors/backend/config.py` — `set_backend` | Reassignable at any time | **Changed.** T7's lifecycle, including what initialization means (Q1). |
| `tensors/backend/config.py` — `use_backend` | Scoped override | Unchanged (T8), but see Q2. |
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

## 8. Open questions

Each records the alternatives, what follows from them, and a recommendation.
**None is settled by this document.**

### Q1 — The scope of the one-time rule

What marks a process as initialized, and what is the unit the rule applies to?

| Option | Consequence |
| --- | --- |
| **A. First tensor storage constructed** | Matches T7 as written. A program may select freely during import and configuration, and is fixed from its first tensor. Requires one process-wide flag. |
| **B. First operation executed** | More permissive: tensors may be built, then the backend chosen. But tensors built before the choice already have a backend, so T3 makes them mismatched — the rule would not prevent the failure it exists to prevent. |
| **C. Per-thread rather than per-process** | Allows a worker per backend. Multiplies the lifecycle by the thread model and interacts with `use_backend`'s context-local override in ways not thought through. |

**Recommended: A**, with the flag reset only by an explicit test-support hook.
B does not achieve the rule's purpose; C is a larger design and should be a
separate question if multi-backend processes are ever wanted.

### Q2 — `use_backend` against the one-time rule

T7 fixes the selection; T8 lets a block change it. They must be reconciled.

| Option | Consequence |
| --- | --- |
| **A. `use_backend` is exempt** | Scoped selection keeps working. A block may construct tensors on a second backend, so one process holds tensors of two backends — legal under T3, and every cross-use is an error under T6. This is the status quo plus enforcement. |
| **B. `use_backend` is restricted to the initialized backend** | The one-time rule holds absolutely; `use_backend` becomes an assertion rather than a selection, and its remaining purpose is unclear. |
| **C. `use_backend` is removed** | Simplest model, largest breaking change. The test suite uses it heavily to exercise all three backends in one process. |

**Recommended: A.** The one-time rule exists to stop a program's tensors
silently disagreeing with its selection; a scoped block that constructs and
consumes its own tensors does not create that problem, and T6 catches it if it
escapes. Note that A means T7's "one-time" governs the *process default*, not
every selection — T7 should be worded to say so once this is settled.

### Q3 — Construction from a Tensor of another backend

`Tensor(other)` where `other` belongs to a different backend.

| Option | Consequence |
| --- | --- |
| **A. Raise** | Consistent with T6: construction is not a transfer mechanism. The caller has no way to move data, which makes Q8 pressing. |
| **B. Copy to the active backend** | Convenient, and a natural reading of "construct a tensor here from these values". But it is a cross-backend transfer under an ordinary-looking constructor, which is the pattern T5 removes. |
| **C. Copy the source's backend** | Today's behaviour (3.5). Construction then ignores the selection, contradicting T1. |

**Recommended: A**, paired with a decision on Q8. B is the same invisible
transfer in a different place; C contradicts T1.

### Q4 — Construction from a Storage object

`Tensor(storage)` where the storage belongs to a different backend. The same
three options as Q3 apply, with one difference: a `Storage` is an internal
type, so the public surface is narrower and the compatibility cost of raising
is lower.

**Recommended: A (raise)**, for consistency with Q3.

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

No serialization exists (3.9), so this is a question about future work.

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

**Recommended: defer.** No concrete requirement for it has been stated. Decide
after Q3 is settled and the implementation shows whether the strict model is
usable; adding it later is compatible, removing it later is not.

## Related documents

- [Numerical backends](backends.md) — selection, execution requirements and
  residency **as implemented**.
- [Tensor memory model](memory-model.md) — layout, ownership and the deferred
  view work.
- [Arithmetic semantics](arithmetic-semantics.md) — the numerical contract,
  unaffected by this proposal.
- [Package structure](package-structure.md) — where the affected modules live.
