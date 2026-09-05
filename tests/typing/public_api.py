"""Static contract for the public API; checked by mypy, not executed."""

from typing import Literal

from typing_extensions import assert_type

import tensors as ts
from tensors.storage import Storage


def tensor_from_storage(storage: Storage) -> ts.Tensor:
    return ts.Tensor(storage)


tensor = ts.Tensor([[1.0, 2.0]])
variable = ts.Variable([[1.0, 2.0]])

shape = ts.Shape(2, 3)
strides = ts.Strides.contiguous(shape)
assert_type(shape, ts.Shape)
assert_type(shape.rank, int)
assert_type(shape.size, int)
assert_type(shape[1:], ts.Shape)
assert_type(shape.broadcast_with((1, 3)), ts.Shape)
assert_type(strides, ts.Strides)
assert_type(tensor.shape, ts.Shape)
assert_type(tensor.strides, ts.Strides)
assert_type(tensor.offset, int)
assert_type(tensor.is_contiguous, bool)
assert_type(tensor.contiguous(), ts.Tensor)
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
assert_type(ts.backward(loss), None)
assert_type(ts.grad(loss, variable), ts.Tensor | ts.Variable | None)
assert_type(
    ts.grad(loss, [variable]),
    tuple[ts.Tensor | ts.Variable | None, ...],
)
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

initializer: ts.init.Initializer = ts.init.HeNormal(dtype=ts.float32)
assert_type(initializer((128, 64)), ts.Tensor)
assert_type(ts.init.VarianceScaling()((128, 64)), ts.Tensor)
assert_type(ts.init.XavierUniform()((128, 64)), ts.Tensor)
assert_type(ts.init.XavierNormal()((128, 64)), ts.Tensor)
assert_type(ts.init.HeUniform()((128, 64)), ts.Tensor)
assert_type(ts.init.HeNormal()((128, 64)), ts.Tensor)
assert_type(ts.init.LecunUniform()((128, 64)), ts.Tensor)
assert_type(ts.init.LecunNormal()((128, 64)), ts.Tensor)
assert_type(ts.init.TruncatedNormal()((128, 64)), ts.Tensor)
assert_type(ts.init.Orthogonal()((128, 64)), ts.Tensor)

signal = ts.Tensor([[[1.0, 2.0, 3.0, 4.0]]])
filters = ts.Tensor([[[1.0, -1.0]]])
offsets = ts.Tensor([0.0])
signal_variable = ts.Variable([[[1.0, 2.0, 3.0, 4.0]]])
filters_variable = ts.Variable([[[1.0, -1.0]]])
offsets_variable = ts.Variable([0.0])

assert_type(ts.conv1d(signal, filters), ts.Tensor)
assert_type(ts.conv1d(signal, filters, offsets), ts.Tensor)
assert_type(ts.conv1d(signal_variable, filters), ts.Variable)
assert_type(ts.conv1d(signal, filters_variable), ts.Variable)
assert_type(ts.conv1d(signal, filters, offsets_variable), ts.Variable)
assert_type(
    ts.conv1d(signal, filters, stride=2, padding=1, dilation=2, groups=1),
    ts.Tensor,
)

image = ts.Tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
window = ts.Tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
image_variable = ts.Variable([[[[1.0, 2.0], [3.0, 4.0]]]])
window_variable = ts.Variable([[[[1.0, 0.0], [0.0, 1.0]]]])

assert_type(ts.conv2d(image, window), ts.Tensor)
assert_type(ts.conv2d(image, window, offsets), ts.Tensor)
assert_type(ts.conv2d(image_variable, window), ts.Variable)
assert_type(ts.conv2d(image, window_variable), ts.Variable)
assert_type(ts.conv2d(image, window, offsets_variable), ts.Variable)
assert_type(ts.math.conv2d(image, window), ts.Tensor)
assert_type(
    ts.conv2d(
        image,
        window,
        stride=(1, 2),
        padding=(1, 0),
        dilation=(2, 1),
        groups=1,
    ),
    ts.Tensor,
)

volume = ts.ones((1, 1, 3, 3, 3))
volume_kernel = ts.ones((1, 1, 2, 2, 2))
volume_variable = ts.Variable(volume)
volume_kernel_variable = ts.Variable(volume_kernel)

assert_type(ts.conv3d(volume, volume_kernel), ts.Tensor)
assert_type(ts.conv3d(volume_variable, volume_kernel), ts.Variable)
assert_type(ts.conv3d(volume, volume_kernel_variable), ts.Variable)
assert_type(ts.conv3d(volume, volume_kernel, offsets_variable), ts.Variable)
assert_type(ts.math.conv3d(volume, volume_kernel), ts.Tensor)
assert_type(
    ts.conv3d(
        volume,
        volume_kernel,
        stride=(1, 2, 1),
        padding=(1, 0, 1),
        dilation=(1, 1, 2),
        groups=1,
    ),
    ts.Tensor,
)