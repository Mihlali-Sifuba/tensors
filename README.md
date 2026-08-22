# tensors

<p align="center">
  <strong>A small tensor and automatic-differentiation engine, written in pure Python.</strong>
</p>

<p align="center">
  Learn how tensor operations, computation graphs, gradients, and optimizers work by building with them directly.
</p>

<p align="center">
  <a href="https://pypi.org/project/ms-tensors/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/ms-tensors?color=3775A9"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white">
  <img alt="Backend" src="https://img.shields.io/badge/backend-pure%20Python-7A3E9D">
  <img alt="Status" src="https://img.shields.io/badge/status-experimental-F59E0B">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-D22128"></a>
</p>

---

`tensors` is a compact numerical-computing project for exploring the machinery behind modern machine-learning frameworks. It includes multidimensional tensors, broadcasting, reverse-mode automatic differentiation, reusable computation graphs, higher-order derivatives, common mathematical functions, losses, and optimizers—all without hiding the implementation behind a large numerical backend.

> [!NOTE]
> This project is currently an educational, experimental implementation. It prioritizes clarity and correctness over production-scale performance.

## Highlights

- Multidimensional `Tensor` values with shapes, indexing, slicing, broadcasting, and dtypes
- Trainable `Variable` values and reverse-mode automatic differentiation
- First-order gradients plus Jacobians, Hessians, and higher-order derivative graphs
- Class-based and function-based `Graph` models with automatic parameter discovery
- Linear algebra including matrix multiplication, dot products, outer products, transposes, and norms
- Neural-network functions including ReLU, sigmoid, tanh, softmax, and softplus
- Stable cross-entropy and binary cross-entropy losses
- SGD, Adam, and RMSprop optimizers
- A broad `unittest` test suite covering tensors, graphs, autograd, math, and optimizers

## Installation

Install the latest release from [PyPI](https://pypi.org/project/ms-tensors/):

```powershell
python -m pip install ms-tensors
```

The distribution is named `ms-tensors`, while the Python package is imported as
`tensors`. It supports Python 3.10 and later and has no third-party runtime
dependencies.

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
| Autograd | `backward`, `grad`, `gradcheck`, `jacobian`, `hessian` |
| Graphs | `Graph`, nested models, function decorators, parameter discovery |
| Linear algebra | `dot`, `matmul`, `outer`, `transpose`, `norm` |
| Reductions | `sum`, `mean`, `min`, `max`, `std`, `logsumexp` |
| Activations | `relu`, `sigmoid`, `tanh`, `softplus`, `softmax`, `log_softmax` |
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
python -m unittest discover -s tests
```

## Project structure

```text
tensors/
├── tensors/
│   ├── graph/       # computation graphs and automatic differentiation
│   ├── linalg/      # linear-algebra operations
│   ├── math/        # reductions, activations, losses, and shape operations
│   ├── ops/         # primitive differentiable operations
│   ├── optim/       # SGD, Adam, and RMSprop
│   ├── tensor.py    # tensor storage and core behavior
│   └── variable.py  # differentiable tensor values
├── examples/        # runnable demonstrations
└── tests/           # automated test suite
```

## Roadmap

- Introduce an optional NumPy backend
- Expand neural-network building blocks
- Add performance benchmarks and user documentation

## Contributing

Issues, focused bug reports, tests, and small improvements are welcome. Before submitting a change, run the complete test suite and include tests for corrected or newly introduced behavior.

Clone the repository for local development:

```powershell
git clone https://github.com/Mihlali-Sifuba/tensors.git
cd tensors
python -m unittest discover -s tests
```

## License

This project is licensed under the [Apache License 2.0](LICENSE). You may use,
modify, and distribute it under the terms of that license.

---

<p align="center">
  Built to make tensor internals easier to inspect, understand, and improve.
</p>
