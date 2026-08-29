"""Static contract for the public API; checked by mypy, not executed."""

from typing import Literal

from typing_extensions import assert_type

import tensors as ts
from tensors.storage import Storage


def tensor_from_storage(storage: Storage) -> ts.Tensor:
    return ts.Tensor(storage)


tensor = ts.Tensor([[1.0, 2.0]])
variable = ts.Variable([[1.0, 2.0]])

assert_type(tensor.shape, tuple[int, ...])
assert_type(tensor.item(), int | float)
assert_type(variable.data, ts.Tensor)
assert_type(variable.requires_grad, bool)

assert_type(tensor + tensor, ts.Tensor)
assert_type(tensor + variable, ts.Variable)
assert_type(variable * tensor, ts.Variable)
assert_type(ts.pow(variable, 2.0), ts.Variable)

assert_type(ts.sin(tensor), ts.Tensor)
assert_type(ts.sin(variable), ts.Variable)
assert_type(ts.sum([1.0, 2.0]), ts.Tensor)
assert_type(ts.sum(variable), ts.Variable)
assert_type(ts.reshape(variable, (2, 1)), ts.Variable)
assert_type(ts.softmax(variable), ts.Variable)

assert_type(ts.dot(tensor, ts.transpose(tensor)), ts.Tensor)
assert_type(ts.dot(variable, ts.transpose(tensor)), ts.Variable)

assert_type(
    ts.available_backends(),
    tuple[Literal["python", "numpy", "cuda"], ...],
)
assert_type(ts.get_backend(), Literal["python", "numpy", "cuda"])
assert_type(ts.set_backend("python"), None)
with ts.use_backend("numpy"):
    assert_type(ts.get_backend(), Literal["python", "numpy", "cuda"])
with ts.use_backend("cuda"):
    assert_type(ts.get_backend(), Literal["python", "numpy", "cuda"])
assert_type(ts.maximum(variable, tensor), ts.Variable)
assert_type(ts.where(ts.greater(tensor, 0.0), variable, tensor), ts.Variable)
assert_type(ts.concat([variable, variable]), ts.Variable)
assert_type(ts.stack([tensor, tensor]), ts.Tensor)

targets = ts.Tensor([1], dtype=ts.int64)
loss = ts.cross_entropy(variable, targets)
assert_type(loss, ts.Variable)
assert_type(ts.grad(loss, variable), ts.Tensor | ts.Variable | None)
assert_type(ts.jacobian(loss, variable), ts.Tensor | ts.Variable)
assert_type(ts.hessian(loss, variable), ts.Tensor | ts.Variable)

assert_type(ts.random.seed(42), None)
assert_type(ts.random.uniform((2, 3)), ts.Tensor)
assert_type(ts.random.normal((2, 3), dtype=ts.float32), ts.Tensor)
assert_type(ts.random.randint((2, 3), 0, 10), ts.Tensor)

assert_type(ts.init.variance_scaling((128, 64)), ts.Tensor)
assert_type(ts.init.xavier_uniform((128, 64)), ts.Tensor)
assert_type(ts.init.xavier_normal((128, 64)), ts.Tensor)
assert_type(ts.init.he_uniform((128, 64)), ts.Tensor)
assert_type(ts.init.he_normal((128, 64)), ts.Tensor)
assert_type(ts.init.lecun_uniform((128, 64)), ts.Tensor)
assert_type(ts.init.lecun_normal((128, 64)), ts.Tensor)
assert_type(ts.init.truncated_normal((128, 64)), ts.Tensor)
assert_type(ts.init.orthogonal((128, 64)), ts.Tensor)
