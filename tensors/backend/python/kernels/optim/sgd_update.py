"""Apply SGD using Python arithmetic and its intermediate cast semantics."""

from tensors.tensor import Tensor
from tensors.dtype import result_dtype
from tensors.backend.python.kernels.arithmetic.multiply import multiply
from tensors.backend.python.kernels.arithmetic.subtract import subtract


def sgd_update(parameter, gradient, learning_rate):
    """Step each parameter against its gradient."""
    dtype = result_dtype(gradient.dtype, learning_rate)
    scaled = Tensor._from_owned_storage(
        multiply(gradient, learning_rate, dtype=dtype, output_shape=gradient.shape),
        dtype=dtype,
        shape=gradient.shape,
    )
    return subtract(
        parameter,
        scaled,
        dtype=result_dtype(parameter.dtype, scaled),
        output_shape=parameter.shape,
    )
