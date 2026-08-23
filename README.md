# tensors

<p align="center">
  <strong>A small tensor and automatic-differentiation engine with a transparent Python reference implementation.</strong>
</p>

<p align="center">
  Learn how tensor operations, computation graphs, gradients, and optimizers work by building with them directly.
</p>

<p align="center">
  <a href="https://pypi.org/project/ms-tensors/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/ms-tensors?color=3775A9"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Backend" src="https://img.shields.io/badge/backend-Python%20%7C%20NumPy%20%7C%20CUDA-7A3E9D">
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-F59E0B">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-D22128"></a>
</p>

---

`tensors` is a compact numerical-computing project for exploring the machinery behind modern machine-learning frameworks. It includes multidimensional tensors, broadcasting, reverse-mode automatic differentiation, reusable computation graphs, higher-order derivatives, common mathematical functions, losses, and optimizers. A transparent pure-Python implementation defines the behaviour, while optional NumPy and CUDA kernels can accelerate supported operations without changing the mathematical API.

> [!NOTE]
> This project is currently an educational, experimental implementation. It prioritizes clarity and correctness over production-scale performance.

## API philosophy

`tensors` starts from a simple premise: a numerical package should let users
state the mathematics directly. The primary interface is the expression itself:

```python
prediction = inputs @ weight + bias
loss = ts.mean((prediction - target) ** 2.0)
```

Python operators represent algebra, common mathematical functions are available
from the root `ts` namespace, and `Tensor` and `Variable` use the same
expression vocabulary.

A function can be represented in several ways: as a formula, as values, as a
plotted graph, or as a sequence of computations. A computational graph is that
last representation. Its nodes and edges describe how inputs flow through
operations to produce outputs. For differentiable tensor functions, the same
structure records the dependencies that reverse-mode automatic differentiation
follows when applying the chain rule.

`Graph` therefore represents a function; it is not an additional model-building
procedure placed around one. Defining `forward` defines the function, and
calling the `Graph` both evaluates that function eagerly and records its fresh
computational representation. Parameters remain ordinary `Variable`
attributes. There is no separate layer-registration, graph-construction, or
compilation ceremony required before stating the computation.

For example, consider the affine function

$$
f_{\theta}(x) = xW + b, \qquad \theta = \{W, b\}.
$$

Use a `Graph` subclass when the function has explicit state or child graphs:

```python
class Linear(ts.Graph):
    def __init__(self) -> None:
        super().__init__()
        self.weight = ts.Variable([[0.5]], name="weight")
        self.bias = ts.Variable([0.0], name="bias")

    def forward(self, inputs):
        return inputs @ self.weight + self.bias
```

For a small function, the decorator form represents the same idea directly:

```python
weight = ts.Variable([[0.5]], name="weight")
bias = ts.Variable([0.0], name="bias")


@ts.Graph
def linear(inputs):
    return inputs @ weight + bias
```

Both forms produce callable `Graph` objects, discover their trainable
parameters, and record the computation expressed by the function body.

Training is also an algorithm, so the training loop remains explicit code:

```python
for _ in range(steps):
    prediction = model(inputs)
    loss = ts.mean((prediction - target) ** 2.0)

    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()
```

Those visible steps are part of the idea being expressed: evaluate the
function, measure the loss, differentiate it, and update the parameters.

Organised subpackages are available for discovery and advanced use, but ordinary
mathematical code should not require users to import a separate hierarchy of
layers, modules, or parameter-registration abstractions. An abstraction belongs
in the public API when it makes the mathematics clearer or enables necessary
behaviour—not when it merely adds ceremony around an expression Python already
represents well.

The numerical backend follows the same principle. Backend selection is
application configuration, not a second expression language: tensors, graphs,
gradients, and training loops are written identically under the Python, NumPy,
and CUDA implementations.

## Highlights

- Multidimensional `Tensor` values with shapes, indexing, slicing, broadcasting, and dtypes
- Trainable `Variable` values and reverse-mode automatic differentiation
- First-order gradients plus Jacobians, Hessians, and higher-order derivative graphs
- Class-based and function-based `Graph` models with automatic parameter discovery
- Linear algebra including matrix multiplication, dot products, outer products, transposes, and norms
- Runtime-selectable Python, NumPy, and CUDA numerical backends
- Neural-network functions including ReLU, sigmoid, tanh, softmax, and softplus
- Stable cross-entropy and binary cross-entropy losses
- SGD, Adam, and RMSprop optimizers
- Inline type information distributed through the standard `py.typed` marker
- A broad `unittest` test suite covering tensors, graphs, autograd, math, and optimizers

## Installation

Install the latest release from [PyPI](https://pypi.org/project/ms-tensors/):

```powershell
python -m pip install ms-tensors
```

The distribution is named `ms-tensors`, while the Python package is imported as
`tensors`. It supports Python 3.10 and later. The default Python backend has no
third-party runtime dependencies.

Install the optional NumPy backend into the same environment with:

```powershell
python -m pip install "ms-tensors[numpy]"
```

For an NVIDIA GPU, install one CuPy build matching the CUDA generation supported
by the installed driver:

```powershell
python -m pip install "ms-tensors[cuda12]"
# or
python -m pip install "ms-tensors[cuda13]"
```

Installed backends coexist in the same environment. Python is the default;
select an accelerated backend once before expressing the computation:

```python
import tensors as ts

ts.set_backend("numpy")
# or: ts.set_backend("cuda")
```

Use `ts.use_backend(...)` for a scoped override or set `TENSORS_BACKEND` for a
script. See [Numerical backends](docs/backends.md) for selection, fallback, and
concurrency behaviour.

## Quick start

```python
import tensors as ts

x = ts.Tensor([[1.0, 2.0], [3.0, 4.0]])
y = ts.Tensor([[2.0, 0.0], [1.0, 2.0]])

print(x + y)
print(x @ y)
print(ts.mean(x))
```

## Automatic differentiation

Create trainable variables, build an expression, and differentiate it:

```python
import tensors as ts

x = ts.Variable([2.0, 3.0], name="x")
loss = ts.sum(x ** 3.0)

gradient = ts.grad(loss, x)

print("loss:", loss.data.tolist())       # [35.0]
print("gradient:", gradient.tolist())   # [12.0, 27.0]
```

Higher-order derivatives use the same public API:

```python
hessian = ts.hessian(loss, x)
print(hessian.tolist())  # [12.0, 0.0, 0.0, 18.0]
```

## Build a model

Subclass `Graph` and store trainable variables as attributes. The graph discovers parameters inside the model automatically.

```python
import tensors as ts


class Linear(ts.Graph):
    def __init__(self) -> None:
        super().__init__()
        self.weight = ts.Variable([[0.5], [-0.5]], name="weight")
        self.bias = ts.Variable([0.0], name="bias")

    def forward(self, inputs):
        return inputs @ self.weight + self.bias


model = Linear()
inputs = ts.Tensor([[2.0, 1.0]])
prediction = model(inputs)

print(prediction.data.tolist())
print([parameter.name for parameter in model.parameters()])
```

Each call executes `forward` eagerly and records a fresh computation. The
latest outputs, nodes, edges, and computations are available for inspection on
the calling thread; `Graph` does not replay a cached trace on later calls.

Training follows a familiar loop:

```python
optimizer = ts.optim.SGD(model.parameters(), learning_rate=0.05)
target = ts.Tensor([[1.0]])

for _ in range(100):
    prediction = model(inputs)
    loss = ts.mean((prediction - target) ** 2.0)

    optimizer.zero_grad()
    ts.backward(loss)
    optimizer.step()
```

## Public API at a glance

| Area | Available functionality |
| --- | --- |
| Core | `Tensor`, `Variable`, dtypes, indexing, slicing, casting, broadcasting |
| Backends | `available_backends`, `get_backend`, `set_backend`, `use_backend` |
| Creation | `zeros`, `ones`, `full`, `eye`, `arange`, `linspace` |
| Autograd | `backward`, `grad`, `gradcheck`, `jacobian`, `hessian` |
| Graphs | `Graph`, nested models, function decorators, parameter discovery |
| Linear algebra | `dot`, `matmul`, `outer`, `transpose`, `norm` |
| Reductions | `sum`, `prod`, `mean`, `variance`, `std`, `min`, `max`, `argmin`, `argmax`, `logsumexp` |
| Elementwise | `abs`, `sign`, `clip`, `minimum`, `maximum`, comparisons, `where` |
| Trigonometry | `sin`, `cos`, `tan`, `arcsin`, `arccos`, `arctan` |
| Hyperbolic | `sinh`, `cosh`, `tanh`, `arcsinh`, `arccosh`, `arctanh` |
| Activations | `relu`, `sigmoid`, `softplus`, `softmax`, `log_softmax` |
| Shape operations | `reshape`, `stack`, `concat` |
| Losses | `cross_entropy`, `binary_cross_entropy` |
| Optimizers | `SGD`, `Adam`, `RMSprop` |

## Examples

The [`examples`](examples) directory contains runnable demonstrations in a suggested learning order:

1. `computation_forward.py` — inspect and replay a simple computation
2. `graph_structure.py` — explore nodes, edges, and graph state
3. `higher_order_gradients.py` — calculate first, second, and third derivatives
4. `multilayer_perceptron.py` — train a two-layer neural network with Adam
5. `mlp_threads.py` — compare independent MLP training across threads

Run one from the project root:

```powershell
python -m examples.higher_order_gradients
```

## Run the tests

The test suite uses Python's standard-library test runner:

```powershell
python -m unittest discover -s tests -t .
```

Check the public static typing contract with:

```powershell
python -m pip install -e ".[typing]"
python -m mypy
```

## Run the benchmarks

A dependency-free benchmark suite tracks operations from raw provider kernels
through storage, the public API, graph tracing, automatic differentiation, and
complete training phases. The compact `core` suite compares all available
numerical backends by default. Run its short development configuration with:

```powershell
python -m benchmarks --quick
```

Use `python -m benchmarks` for a longer core run, or target an attribution suite
such as `python -m benchmarks --backend accelerated --suite scaling`. Write a
machine-readable report with `--output benchmark-results.json`. See the
[`benchmarks` guide](benchmarks/README.md) for the full case matrix, backend
eligibility, and measurement methodology.

## Project structure

```text
tensors/
├── tensors/
│   ├── backend/           # backend selection and optional kernels
│   ├── storage/           # Python, NumPy, and CUDA native storage
│   ├── graph/             # computation graphs and automatic differentiation
│   ├── linalg/            # linear-algebra operations
│   ├── math/              # reductions, activations, losses, and shape operations
│   ├── ops/               # primitive differentiable operations
│   ├── optim/             # SGD, Adam, and RMSprop
│   ├── creation.py        # zeros, ones, ranges, and identity matrices
│   ├── tensor.py          # tensor storage and core behavior
│   └── variable.py        # differentiable tensor values
├── examples/        # runnable demonstrations
├── benchmarks/      # repeatable local performance baselines
└── tests/           # automated test suite
```

## Roadmap

- Expand neural-network building blocks

## Contributing

Issues, focused bug reports, tests, and small improvements are welcome. Before submitting a change, run the complete test suite and include tests for corrected or newly introduced behavior.

Clone the repository for local development:

```powershell
git clone https://github.com/Mihlali-Sifuba/tensors.git
cd tensors
python -m unittest discover -s tests -t .
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). You may use,
modify, and distribute it under the terms of that license.

---

<p align="center">
  Built to make tensor internals easier to inspect, understand, and improve.
</p>
