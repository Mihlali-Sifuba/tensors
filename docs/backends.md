# Numerical backends

The public API expresses tensor mathematics independently of its numerical
implementation. Backend selection changes how supported kernels are calculated;
it does not create a different tensor type or alter how graphs, gradients, and
training loops are written.

## Installation

The Python backend has no third-party runtime dependencies and is always
available. Install the optional NumPy backend with the same package:

```powershell
python -m pip install "ms-tensors[numpy]"
```

Both backends then coexist in the same environment. Tensors retain canonical
`array.array` storage, so the same tensor and recorded computation can be used
with either backend.

## Selection

Python is the default backend. Select NumPy once for an application:

```python
import tensors as ts

ts.set_backend("numpy")
```

Inspect backend state with:

```python
ts.available_backends()
ts.get_backend()
```

Use a scoped override for tests, comparisons, or one complete training step:

```python
with ts.use_backend("numpy"):
    prediction = model(inputs)
    loss = ts.mean((prediction - targets) ** 2.0)
    ts.backward(loss)
    optimizer.step()
```

Scoped overrides are context-local and restore the previous backend even when an
exception is raised. Set the process default before starting worker threads;
new threads use that process default rather than inheriting a temporary override.

For scripts and CI, select the initial backend through the environment:

```powershell
$env:TENSORS_BACKEND = "numpy"
python train.py
```

Valid selections are `python`, `numpy`, and `auto`. Auto mode selects NumPy when
it is installed and otherwise uses Python. An explicit unavailable or unknown
backend raises an error rather than silently changing implementations.

## Kernel coverage

The NumPy backend accelerates the numerical surface without introducing a
second public API. Kernel families include:

- broadcasting arithmetic, power, negation, casting, slicing, and slice scatter;
- unary mathematical functions and their first-order gradients;
- sum, mean, variance, standard deviation, product, norm, extrema, and
  arg-extrema reductions;
- softmax, log-softmax, log-sum-exp, cross-entropy, and binary cross-entropy;
- comparisons, `where`, clipping, and elementwise minimum and maximum;
- floating-point `dot`, `matmul`, outer products, transposes, concatenation,
  and stacking;
- constant, range, evenly spaced, and identity-like tensor construction; and
- fused SGD, Adam, and RMSprop parameter updates.

Unary, normalization, loss, outer-product, extrema, clipping, and selection
families include dedicated first-order kernels. Other gradient rules compose
already-dispatched primitives or structural slices. Broadcast-gradient
reductions are fused so an elementwise derivative can multiply and reduce
without constructing another Python-level broadcast traversal.

Integer matrix products and floating-point edge cases that require the
reference implementation's stable summation, scaled moments, exact
cancellation, or boundary rules fall back to the Python kernel.

Backend selection expresses a preference rather than forcing every operation
through NumPy. Internal dispatch keeps very small elementwise operations and
matrix products on the Python implementation when NumPy setup would cost more
than the numerical work. Current local crossover benchmarks use 32 output
elements for elementwise kernels, eight input elements for reductions, and 32
multiply-accumulate work units for matrix products.

Graph recording and automatic differentiation remain backend-agnostic. Primitive
operations used while constructing or differentiating a graph use the active
backend, and replaying a recorded computation uses the backend active for that
replay.

## Behaviour contract

The Python implementation defines backend-independent shape, dtype, error, and
automatic-differentiation behaviour. NumPy kernels preserve that contract. Exact
integer results and structural behaviour must match; floating-point results are
expected to agree within dtype-appropriate tolerances.

NumPy arrays are not exposed through this initial backend. Public array
interoperability requires separate copy, shared-memory, mutation, and versioning
semantics and is therefore outside the backend-selection contract.
