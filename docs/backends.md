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

The NumPy backend provides kernels for every primitive operation in `tensors.ops`:

- broadcasting addition, subtraction, multiplication, and division;
- elementwise power, including scalar and reverse forms;
- negation;
- tensor indexing and slicing;
- differentiable slice scatter;
- dtype casting;
- range-safe division-denominator gradients; and
- power gradients with respect to bases and exponents.

It also accelerates floating-point `dot` and `matmul`, including vector, matrix,
batched, and broadcasted matrix products, together with `sum`, `mean`,
`variance`, and Euclidean `norm` reductions. Integer matrix products and
floating-point edge cases that require the reference implementation's stable
summation fall back to the Python kernel.

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
