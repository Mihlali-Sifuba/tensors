"""Matrix multiplication and its differentiation rule."""

from typing import Any, List

from ..tensor import Tensor
from ..ops._utils import result_dtype


def _dot_impl(a: Tensor, b: Tensor) -> Tensor:
    """Actual matrix multiplication for the currently supported 2D inputs."""
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("Dot product only supported for 2D tensors")
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Cannot multiply {a.shape} with {b.shape}: inner dimensions must match"
        )

    dtype = result_dtype(a.dtype, b)
    result = dtype.make_array([])
    a_cols = a.shape[1]
    b_cols = b.shape[1]

    for i in range(a.shape[0]):
        for j in range(b_cols):
            total = 0
            for k in range(a_cols):
                total += a._data[i * a_cols + k] * b._data[k * b_cols + j]
            result.append(total)

    return Tensor(result, dtype=dtype, shape=(a.shape[0], b.shape[1]))


def _transpose_impl(t: Tensor) -> Tensor:
    """Transpose a 2D tensor for matrix operations and backward passes."""
    if t.ndim != 2:
        raise ValueError("Transpose only supported for 2D tensors")
    new_data = t.dtype.make_array([])
    for j in range(t.shape[1]):
        for i in range(t.shape[0]):
            new_data.append(t._data[i * t.shape[1] + j])
    return Tensor(new_data, dtype=t.dtype, shape=(t.shape[1], t.shape[0]))


class Dot:
    """2D matrix multiplication with a reverse-mode gradient rule."""

    forward = staticmethod(_dot_impl)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        a, b = inputs
        og = grad
        if og.ndim == 1:
            og = Tensor(og._data, shape=(1, og.shape[0]))
        return [_dot_impl(og, _transpose_impl(b)), _dot_impl(_transpose_impl(a), og)]


def dot(a: Any, b: Any) -> Any:
    """Multiply two 2D Tensors or differentiable Variables."""
    from ..autograd.variable import Variable, dot as variable_dot

    if isinstance(a, Variable) or isinstance(b, Variable):
        return variable_dot(a, b)
    return Dot.forward(a, b)


__all__ = ["Dot", "dot", "_transpose_impl"]
