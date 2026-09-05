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
to physical storage, and `Storage` owns the flat provider buffer. Optional
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

## Kernel coverage and fallback

NumPy and CUDA share kernels for broadcasting arithmetic, unary mathematics,
reductions, normalization, losses, selection, layout operations, tensor
construction, linear algebra, convolution, gradients, and fused optimizer
updates. Convolution and its VJPs use bounded matrix-product tiles, so grouped
and dilated kernels stay device-resident without materializing an unbounded
receptive-field matrix. Float32 convolution remains float32 on accelerated
providers; mixed inputs use the public result dtype.

The Python implementation defines shape, dtype, error, and differentiation
semantics. Optional kernels return to that implementation for edge cases that
need stable reference algorithms or exact Python integer intermediates. CuPy has
no Python object dtype, so exact integer operations currently use the Python
path; floating-point kernels remain device-resident.

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
python -m benchmarks --backend accelerated --suite provider
python -m benchmarks --backend accelerated --suite scaling
python -m benchmarks --backend accelerated --suite storage
python -m benchmarks --backend accelerated --suite graph --match width-100000
python -m benchmarks --backend accelerated --suite optimizer
python -m benchmarks --backend accelerated --suite convolution
```

The `provider` suite separates NumPy or CuPy time from internal kernel guards,
`storage` exposes transfer and materialization costs, and `scaling` shows the
size at which an accelerator begins to repay its fixed overhead. CUDA timings
include stream synchronization, so they represent completed device work. See
the [benchmark guide](../benchmarks/README.md) for the complete methodology.

## Behaviour contract

Exact integer results and structural behaviour must match the Python reference.
Floating-point results are expected to agree within dtype-appropriate
tolerances. Changing a backend is an execution choice, not a change to the
mathematical API.
