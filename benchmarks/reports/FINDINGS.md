# Structural findings from reading the implementation

These are properties of the code at commit `04e8dd25`, established by reading
it and confirmed by direct probes. They are separate from the benchmark
measurements, and they are what the measurements were designed to price. Each
one names the file and the reason it matters.

## 1. Every floating-point kernel computes in float64, whatever the tensor dtype

`tensors/backend/kernels/core.py`, `_operand`:

```python
working_dtype = numpy.float64 if dtype.kind == "floating" else object
return numpy.asarray(result, dtype=working_dtype)
```

A float32 tensor is widened to float64 before the arithmetic and narrowed
afterwards. The stated reason is to match the Python reference
implementation's accumulation precision.

Consequence: float32 cannot be faster than float64 through this path, and on a
device whose float32 throughput far exceeds its float64 throughput the
ordering inverts.

## 2. Narrowing a float32 result adds three full passes and a host barrier

`tensors/backend/kernels/core.py`, `_storage`:

```python
target_dtype = numpy.dtype(dtype.name)
if target_dtype.itemsize < numpy.dtype(numpy.float64).itemsize:
    finite = numpy.isfinite(flattened)
    outside_range = numpy.abs(flattened) > numpy.finfo(target_dtype).max
    if bool(numpy.any(finite & outside_range)):
        return None
```

The guard runs only when the target is narrower than float64 — that is, only
for float32. It adds `isfinite`, `abs`, a comparison and a reduction over the
whole result, and the `bool(...)` is a device-to-host read, so on CUDA every
float32 elementwise operation synchronizes.

float64 skips this branch entirely, which is why the two dtypes take
measurably different routes rather than differing only in width.

## 3. CUDA integer arithmetic runs on the host, in Python

`tensors/backend/kernels/core.py`, `_operand` and `_storage`:

```python
if _array_kind(numpy) == "cuda" and dtype.kind == "integer":
    raise TypeError("CUDA integer kernels require the Python fallback")
...
if dtype.kind == "integer":
    if backend == "cuda":
        return None
```

CuPy has no object dtype with Python's unbounded integer semantics, so integer
kernels decline on CUDA and the caller runs the reference implementation. The
result is host storage: an integer operation under the CUDA backend neither
executes on the device nor leaves its values there. Confirmed by probe —
`(int64 + int64)._storage.kind == "python"` under `use_backend("cuda")`.

## 4. The reduction stability guard never takes its fast path on CUDA

`tensors/backend/kernels/reductions/reduction_ops.py`:

```python
ordinary = False
if get_backend() == "numpy":
    minimum = numpy.min(values, axis=axis, keepdims=True)
    maximum = numpy.max(values, axis=axis, keepdims=True)
    ordinary = bool(numpy.all(...))
if ordinary:
    result = direct
else:
    scaled = _scaled_sum(values, axis, numpy)
    safe = _summation_guard(values, numpy, axes=axis, keepdims=True)
    ...
    if not bool(valid):
        return None
```

The fast path is gated on `get_backend() == "numpy"`, so a CUDA reduction
always runs the full guard: `_scaled_sum` (max, abs, where, divide, sum,
multiply, where) plus `_summation_guard` (min, max, abs, comparison, any,
where, min, max, all) plus `all(isfinite(direct))`, `where`, `any(isnan)`, and
a `bool(...)`.

On NumPy the fast path is available but is itself not free: it adds `min`,
`max`, three `isfinite` calls and an `all` over the values, and it only
applies when the data is finite and same-sign — which is why same-sign and
mixed-sign data are benchmarked separately.

`variance`, `std` and `norm` have no fast path at all and each end in
`bool(valid)`.

## 5. Matrix multiplication reads its result back to the host

`tensors/backend/kernels/linalg/matmul_ops.py`:

```python
result = numpy.matmul(left_array, right_array)
if not bool(numpy.all(numpy.isfinite(result))):
    ...
```

Every matmul performs an `isfinite` pass over the output and a `bool(...)`, so
on CUDA every matrix product synchronizes — including one inside a training
step, where the barrier prevents the next layer's launches from overlapping
this layer's execution.

## 6. No public operation returns a view; every shape operation copies

Confirmed by probe. `reshape`, `transpose`, slicing, `astype` and `clone` all
return independently owned, compact storage, none of it shared with the
source:

| operation | contiguous | compact | shares storage |
| --- | --- | --- | --- |
| `reshape` | yes | yes | no |
| `transpose` | yes | yes | no |
| slice | yes | yes | no |
| `clone` | yes | yes | no |
| `astype` | yes | yes | no |
| `contiguous` (already contiguous) | yes | yes | **yes** |

`Tensor` does support non-compact layouts — `strides`, `offset` and
`is_contiguous` are all real — but only `Tensor._from_metadata` produces one.
So "view versus materialized" is not a choice the public API offers, and the
layout benchmarks reach non-compact tensors through that internal constructor.

## 7. A non-compact layout is gathered one element at a time, in Python

`tensors/tensor.py`, `_logical_storage_indices` and `_logical_storage_for`:

```python
ranges = (range(dimension) for dimension in self.shape)
for coordinates in product(*ranges):
    yield coordinates_to_storage_index(
        coordinates, self.shape, self.strides, self.offset,
    )
...
indices = list(self._logical_storage_indices())
selected = storage.buffer[indices]
```

Crossing the provider boundary with a non-compact tensor builds a Python list
of every logical index, using `itertools.product` and a per-element index
calculation. This is the same code on all three backends: on CUDA the gather
list is built on the host and then used for fancy indexing on the device.

## 8. `Tensor()` accepts no provider array

`tensors/tensor.py`, `__init__` accepts `Tensor`, `Storage`, `int`, `float`,
`list` and `array.array`, and raises otherwise. A NumPy or CuPy array is not
accepted, so there is no public zero-copy path from provider values into a
Tensor; the supported route is to wrap the buffer in the matching `Storage`
class, which public construction then copies.

Separately, `Tensor(list)` always builds `PythonStorage`, so a Tensor built
from host values under the CUDA backend holds host storage until some kernel
asks for the device representation.

## 9. Reading any value to the host materializes through Python

`tensors/tensor.py`: `item()`, `tolist()` and `__eq__` all go through `_data`,
which is `_logical_storage_for("python")`. On CUDA that is
`cupy.asnumpy(...)`, then `array.frombytes(values.astype(...).tobytes())`,
then a Python-level conversion. `__eq__` compares `tolist()` on both operands,
so comparing two tensors materializes both.

## 10. Every eager `Variable` operation compiles a program

`tensors/variable.py`, `_apply_operation`:

```python
operands = tuple(operand.node for operand in inputs)
result_node = get_graph_state().record_operation(operation, operands)
fragment, = Computation.from_nodes((result_node,), boundaries=operands)
fragment.forward()
return result_node.variable
```

One eager `a + b` records a result vertex, an operation vertex and the edges
between them, constructs a `Compiler`, runs a traversal, computes reachability
masks, assigns slots, emits one instruction, resolves views, builds a
`Computation`, allocates its slot table, executes the instruction, binds the
result and captures forward state. The design reason is that eager and graph
execution should take the same path; the cost is that the graph machinery is
paid per operation.

## 11. A persistent leaf accumulates one weak reference per operation

`tensors/graph/node.py`:

```python
def _add_out_edge(self, edge: Edge) -> None:
    self._out_edge_references.append(ref(edge))
```

`_out_edge_references` is appended to on every operation and pruned only when
`_out_edges` is next read. A long-lived leaf — a model parameter — therefore
accumulates one dead weak reference per operation performed on it until
something reads its outgoing edges.

## 12. Gradient accumulation stacks before it sums

`tensors/graph/computation/gradients.py`, `sum_gradient_values`:

```python
if get_backend() != "python" and first.size >= 32:
    return tensor_sum(stack(gradients, axis=0), axis=0)
```

Combining *n* gradient contributions materializes an *n* × size tensor and
then runs a guarded reduction over it. A value consumed in many places — the
shared subexpression case — pays that at its fan-in.
