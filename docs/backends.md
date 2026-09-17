# Numerical backends

The public API expresses tensor mathematics independently of its numerical
implementation. Backend selection changes how supported kernels are calculated;
it does not create another tensor type or change how graphs, gradients, and
training loops are written.

## Installation

The Python backend has no third-party runtime dependencies and is always
available. Optional backends are installed into the same environment:

```powershell
python -m pip install "ms-tensors[numpy]"
python -m pip install "ms-tensors[cuda12]"  # CUDA 12 runtime
python -m pip install "ms-tensors[cuda13]"  # CUDA 13 runtime
```

Install only one CUDA extra. It supplies CuPy and its CUDA runtime libraries;
the machine still needs a compatible NVIDIA driver and CUDA-capable GPU. Python,
NumPy, and the selected CUDA backend can coexist without separate `tensors`
installations.

## Selection

Python is the default. Select an accelerated backend once for an application:

```python
import tensors as ts

ts.set_backend("numpy")
# or
ts.set_backend("cuda")
```

`ts.available_backends()` reports backends that can execute in the current
environment, and `ts.get_backend()` returns the active selection. Use a scoped
override for tests, comparisons, or one complete training step:

```python
with ts.use_backend("cuda"):
    prediction = model(inputs)
    loss = ts.mean((prediction - targets) ** 2.0)
    ts.backward(loss)
    optimizer.step()
```

Scoped overrides are context-local and restore the previous backend even when
an exception is raised. Set the process default before starting worker threads;
new threads use that default rather than inheriting a temporary override.

For scripts and CI, set the initial backend through the environment:

```powershell
$env:TENSORS_BACKEND = "cuda"
python train.py
```

Valid selections are `python`, `numpy`, `cuda`, and `auto`. Auto mode deliberately
selects NumPy when installed and otherwise Python; it does not move work to a GPU
implicitly. Selecting an unavailable or unknown backend raises an error.

## Native storage

Backend selection does not alter the public `Tensor` type, but supported kernels
retain results in their natural internal representation:

- Python uses `array.array`;
- NumPy uses `numpy.ndarray`; and
- CUDA uses device-resident `cupy.ndarray`.

Representations are converted lazily and cached. A Python tensor transferred for
a NumPy or CUDA operation is reused by later operations on that backend, and a
CUDA result stays on the device across a chain of expressions. Public operations
such as `tolist()` materialize host values when needed. An in-place mutation
makes host storage authoritative and invalidates cached native representations,
preventing stale backend data.

These storage classes are internal implementation details, not a second public
array API. Users continue to write ordinary tensor expressions.

Tensor layout metadata remains MS-Tensors' semantic source of truth:
`Shape` defines logical extents, `Strides` and `offset` map logical coordinates
to physical storage, and `Storage` owns the flat backend-native buffer. Optional
backend kernels currently operate on compact arrays. A non-compact internal
layout is gathered in logical order within the same backend before crossing
that boundary, avoiding an unnecessary CUDA-to-host value transfer. Public
layout operations still materialize independent tensors; shared-storage views
are intentionally deferred. See [Tensor memory model](memory-model.md).

Random tensors follow the same residency contract. `ts.random` dispatches to an
MS-Tensors-owned `random.Random`, NumPy `Generator`, or CuPy `RandomState` for
the active backend. Initializers in `ts.init` consume that abstraction, so
large CUDA parameters are sampled directly into device-resident storage.
Calling `ts.random.seed` resets independent backend streams without changing
the provider-global RNGs. See [Parameter initialization](initialization.md) for
the mathematical definitions and reproducibility contract.

## Execution requirements

> **Status: implemented for `+`, `-`, `*` and `/`.** Those four execute on the
> selected backend — including the backend `"auto"` resolved to — or raise
> `BackendOperationUnsupportedError`. No workload-size threshold applies to
> them under any selection. Every other operation still follows the workload
> policy described further down, and may still run the Python reference under
> an explicit selection. The [observability](#observability) mechanism required
> below does not exist.
> Numerical semantics are specified in
> [Arithmetic semantics](arithmetic-semantics.md); this section covers only
> *where* and *whether* an operation executes.

Selecting a backend explicitly is an **execution requirement**, not a
performance preference.

### Vocabulary

Four questions are distinct and are answered separately.

| Term | Question | Answer |
| --- | --- | --- |
| **Availability** | Is the backend usable in this environment? | `ts.available_backends()` |
| **Operation support** | Can this backend execute this operation, at this dtype, conformingly? | per operation and dtype |
| **Fallback** | May another backend execute it instead? | never for `+`, `-`, `*`, `/`; otherwise only under automatic selection |
| **Execution location** | Where did this operation actually run, and where does its result live? | observable; see below |

A backend being *available* does not imply it *supports* every operation.
An operation being supported does not imply it *executed* there — unless
selection was explicit.

### Explicit selection

```python
ts.set_backend("cuda")      # or "numpy", or a scoped use_backend(...)
```

Under explicit selection, for a **supported** operation:

- the operation **must execute using that backend's implementation**;
- it **must not** silently invoke another backend's kernel, including the
  Python arithmetic kernel;
- if it cannot execute correctly there, it **must raise a clear
  unsupported-operation error** naming the operation, the dtype and the
  backend.

The same rule applies to explicit NumPy selection.

An error is the correct outcome when a backend cannot conform. A silent
fallback denies the caller the one thing explicit selection was for: knowing
where the work ran. Where a backend currently cannot conform, and what must be
implemented to close the gap, is recorded in
[Arithmetic semantics §5.4](arithmetic-semantics.md#54-subnormals-gradual-underflow-is-required)
and [§8.4](arithmetic-semantics.md#84-when-a-backend-cannot-conform).

Workload-size policy must not override explicit selection. A small tensor is
still executed on the selected backend; the policy may decide *how*, never
*where*.

This costs something. For `float64` elementwise addition under explicit NumPy
selection, against the Python reference it previously fell back to (minimum of
seven runs of 2000 calls): 1.53x slower at 1 element, 1.47x at 8, 1.14x at 16,
0.94x at 32, 0.31x at 256, 0.02x at 4096. The cost is accepted. A deterministic
rule that a caller can state in one sentence is worth more than a threshold
that makes small-workload behaviour depend on a constant, and any heuristic
worth reintroducing should be argued from benchmark data rather than inherited.

### Automatic selection

```python
ts.set_backend("auto")
```

`"auto"` chooses a backend; it does not choose differently per operation.
It resolves **once, when it is selected**, to NumPy when NumPy is installed
and to Python otherwise. From that point the selection behaves exactly as if
that backend had been named.

For `+`, `-`, `*` and `/` this means automatic selection grants no latitude at
all: no workload-size threshold, no small-tensor case, and no fallback to
another backend. An operation the resolved backend cannot execute conformingly
raises `BackendOperationUnsupportedError`, exactly as under explicit selection.

For operations outside the arithmetic contract, automatic selection may still
use documented workload policies and may fall back to another backend,
**provided the executing path satisfies the numerical contract**. Fallback is
a performance and coverage decision; it is never a licence to produce a
different result.

A fallback taken because a backend's arithmetic differs from Python's is not
legitimate under the new contract. That is the situation the contract removes.

### Observability

Actual execution location must be observable, so that a fallback is visible
rather than inferred from a timing anomaly, and so that a conformance test can
distinguish "computed correctly here" from "computed correctly somewhere else".

**The public API has no such mechanism today.** `ts.available_backends()`,
`ts.get_backend()`, `ts.set_backend()` and `ts.use_backend()` report *selection*
only; nothing reports where an operation ran. Storage type is an indirect and
incomplete proxy: a result in `PythonStorage` after selecting CUDA implies a
fallback, but a device-resident result does not prove every step ran on the
device.

The required API addition is therefore stated rather than invented here:

> **Required.** A way to observe, for a completed operation or a scoped block,
> which backend actually executed it, and whether a fallback occurred. The
> benchmark harness needs this per case; a conformance test needs it per
> operation.

Its exact shape — a context manager, a counter, a structured record, or a
per-result attribute — is an open API decision and is **not** settled by this
document. What is settled is that a fallback must not be silent.

The benchmark suite already records where each case executes and can consume
such a mechanism once it exists; see [`benchmarks/README.md`](../benchmarks/README.md).

### Storage residency and transfers

- An operation's result is stored in its **declared dtype**
  ([Arithmetic semantics §3.4](arithmetic-semantics.md#34-storage-residency-is-not-semantics)).
- Under explicit CUDA selection, a supported operation's result is
  **device-resident**. It must not be materialised on the host as a
  side effect of arithmetic.
- **Intentional host-facing operations transfer by definition.** `tolist()`,
  `item()`, printing, and comparison to a Python value all read values on the
  host; that transfer is the caller's request and is expected.
- **Ordinary arithmetic must not transfer.** A device-to-host copy during
  `a + b`, or a host synchronisation to inspect operand values, is a defect
  under this contract.

The second point has teeth today: the CUDA divide kernel evaluates
`bool(cupy.any(right_array == 0))` to detect a zero denominator, which forces a
host synchronisation on **every** division, and `_storage` does the same to
detect float overflow. Both exist to reproduce Python's exception behaviour,
and both are removed by
[Arithmetic semantics §7.2](arithmetic-semantics.md#72-division-by-zero).

## Kernel coverage and fallback

Each backend owns its own kernels. `tensors/backend/python`, `.../numpy`, and
`.../cuda` each hold one module per operation, so a single operation's Python,
NumPy, and CUDA implementations are three separate files with no shared
numerical body between them. NumPy and CUDA both cover broadcasting arithmetic,
unary mathematics, reductions, normalization, losses, selection, layout
operations, tensor construction, linear algebra, convolution, gradients, and
fused optimizer updates. Convolution and its VJPs use bounded matrix-product
tiles, so grouped and dilated kernels stay device-resident without materializing
an unbounded receptive-field matrix. Float32 convolution remains float32 on
accelerated backends; mixed inputs use the public result dtype.

> **Changing.** The Python implementation currently defines shape, dtype,
> error, and differentiation semantics, and the paragraph below describes that
> arrangement. Under the approved contract it stops being the definition:
> [Arithmetic semantics](arithmetic-semantics.md) is the authority, and the
> declining and falling back described here is removed for arithmetic. See
> [Execution requirements](#execution-requirements) above.

The Python implementation defines shape, dtype, error, and differentiation
semantics for the operations outside the arithmetic contract; for `+`, `-`,
`*` and `/` that role has moved to
[Arithmetic semantics](arithmetic-semantics.md), and their dispatch is
described under [Execution requirements](#execution-requirements) above.
`tensors/backend/dispatch` holds one `execute_*` entry point per
operation; for every operation other than those four, that entry point applies
the workload policy, calls the selected backend's kernel, and runs the Python
reference itself when the policy declines or the kernel does. The four
arithmetic operations share `dispatch/arithmetic/_execution.py`, which consults
no policy: it calls the selected backend and raises if that backend declines.

Their vector-Jacobian products do the same, through `dispatch/_selected.py`.
Where a VJP's computation shares an entry point with something outside the
contract — the broadcast reduction is also used by `power`, `where` and the
losses, and negation is also a forward operation — a second entry point named
`execute_vjp_*` carries the strict policy and the original keeps the old one.
Both call the same kernel; they differ only in what they do when the workload
is small or the kernel declines. Operations therefore receive a result, not a decision: the
choice of fallback belongs to dispatch. An array kernel declines for edge cases
needing stable reference algorithms or exact Python integer intermediates. CuPy
has no Python object dtype, so exact integer operations use the Python path;
floating-point kernels remain device-resident.

Two optional optimizations may still decline to the caller, because their
alternative is ordinary execution rather than a reference kernel: elementwise
fusion falls back to running the steps separately, and a batched optimizer
update falls back to updating each parameter individually.

Small NumPy workloads may use Python when array setup would cost more than the
numerical work. Explicit CUDA selection keeps supported floating-point work on
the device, although launch and synchronization overhead can make small
expressions slower than CPU execution.

Graph recording and automatic differentiation remain backend-agnostic.
Primitives used while tracing, differentiating, or replaying a graph use the
backend active for that operation.

## Performance model

Backend choice is workload-dependent:

- Python is the transparent reference implementation and is useful for
  inspection, exact-integer edge cases, and environments without optional
  dependencies.
- NumPy usually gives the best latency for small and medium CPU workloads. It
  also avoids GPU launch and synchronization costs.
- CUDA is intended for wide tensors, deep replayed computations, large matrix
  operations, and sufficiently batched training work. Small CUDA operations can
  be slower even when their numerical kernel is efficient.

CUDA performs best when values remain device-resident. Chained tensor
expressions, graph replay, native gradients, and optimizer updates preserve
CUDA storage. Calls such as `item()` and `tolist()` intentionally materialize
host values and therefore synchronize or transfer data. Frequent host
inspection inside a training loop can erase the benefit of device execution.

`Computation` resolves graph slots and operations into ordered instructions
once, then executes them with per-pass buffers on forward and backward replay.
Compatible
float32 and float64 elementwise chains—including broadcast tensor arithmetic,
scalar powers, and common unary mathematics—can be fused into one CUDA launch
while retaining the intermediate tensor values required by graph semantics.
Reduction VJPs execute
on the selected optional backend when their stable native path is valid, and
SGD, Adam, and RMSprop batch compatible parameter updates to reduce repeated
dispatch and launch overhead.

These optimizations do not bypass the behaviour contract. Numerically delicate
or unsupported cases still use the stable Python implementation.

Use the benchmark attribution suites instead of one small operation to choose a
backend:

```powershell
python -m benchmarks --backend numpy --backend cuda --suite arithmetic
python -m benchmarks --backend numpy --backend cuda --suite storage
python -m benchmarks --backend numpy --backend cuda --suite graph --match width-100000
python -m benchmarks --backend numpy --backend cuda --suite optimizer
python -m benchmarks --backend numpy --backend cuda --suite convolution
```

Each workload measures the same computation at every depth of the stack, so
one run separates the raw NumPy or CuPy call from the internal kernel guard,
the guard from dispatch, and dispatch from the public operation. Reading the
size curve rather than one point is what shows where an accelerator begins to
repay its fixed overhead; `storage` exposes transfer and materialization costs
separately. CUDA timings are reported three ways — submission, completion, and
device time — so a number is never ambiguous about whether the device had
finished. See the [benchmark guide](../benchmarks/README.md) for the complete
methodology.

## Behaviour contract

**Current behaviour.** Exact integer results and structural behaviour match the
Python reference. Floating-point results are expected to agree within
dtype-appropriate tolerances. Changing a backend is an execution choice, not a
change to the mathematical API.

**Approved target contract.** The declared dtype and a written specification
define an operation's behaviour; no backend is the semantic authority, the
Python backend included. Results, result dtypes and exceptional behaviour are
identical across backends, and for elementary arithmetic that identity is
required **bitwise** rather than within a tolerance, because IEEE 754 requires
those operations to be correctly rounded. The full contract, the breaking
changes it introduces and the conformance requirements are in
[Arithmetic semantics](arithmetic-semantics.md), which is awaiting
implementation and does not describe current behaviour.

Changing a backend remains an execution choice and not a change to the
mathematical API — but under the target contract, *choosing* one explicitly is
a requirement about where execution happens, not only a hint. See
[Execution requirements](#execution-requirements).
