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

## Kernel coverage and fallback

NumPy and CUDA share kernels for broadcasting arithmetic, unary mathematics,
reductions, normalization, losses, selection, layout operations, tensor
construction, linear algebra, gradients, and fused optimizer updates.

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

## Behaviour contract

Exact integer results and structural behaviour must match the Python reference.
Floating-point results are expected to agree within dtype-appropriate
tolerances. Changing a backend is an execution choice, not a change to the
mathematical API.
