"""Addition operation."""

from typing import Union
from tensors.backend import execute_add
from tensors.dtype import convert_scalar, resolve_result_dtype
from tensors.operations.base import Operation
from tensors.operations.manipulation.reshape import reshape
from tensors.operations.reductions.sum import Sum
from tensors.tensor import Tensor

Scalar = Union[int, float]


class Add(Operation):
    """Element-wise addition — forward and backward."""

    __slots__ = ()
    name = "add"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise addition of two tensors or a tensor and a scalar."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        # Two declared dtypes promote; a scalar converts to the tensor's
        # dtype and never widens the result. See section 6.2 and 6.5.
        if isinstance(b, Tensor):
            other = b
            dtype = resolve_result_dtype(a.dtype, b.dtype)
            output_shape = a.shape.broadcast_with(b.shape)
        else:
            other = convert_scalar(b, a.dtype)
            dtype = a.dtype
            output_shape = a.shape
        accelerated = execute_add(
            a, other, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Reduce the upstream gradient to each requested input's shape.

        Addition's derivative with respect to either operand is one, so the
        whole VJP is the broadcast reduction: an operand the forward pass
        stretched takes the sum of the output positions it fed, and an
        operand already of the output's shape takes the gradient unchanged.

        This is the only derivative addition defines. It is written against
        operations rather than against Tensors, so the operands decide what
        the statements mean: given Tensors they calculate, and given
        Variables the same statements record a differentiable graph.
        :class:`~tensors.graph.computation.Computation` calls this method for
        both reverse modes.

        The reduction is asked for on the selected backend, which is the
        contract ``+`` has always reduced under: no workload-size policy, and
        a backend that cannot reduce conformingly says so rather than letting
        Python answer. That choice is the reduction operation's own state, so
        a recorded derivative graph replays under it too.

        An unrequested input costs nothing: its reduction is skipped rather
        than calculated and discarded, and no zero is manufactured in its
        place, because addition's derivative does not depend on its inputs.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        gradients = []
        for operand, requested in zip(inputs, needs_input_grad):
            if not requested:
                gradients.append(None)
                continue
            shape = operand.shape
            if grad.shape == shape:
                gradients.append(grad)
                continue
            if len(shape) > len(grad.shape):
                raise ValueError(
                    f"Cannot reduce gradient shape {grad.shape} to {shape}"
                )
            # A forward broadcast prepends axes and stretches singleton ones.
            # Those are exactly the axes along which one operand value fed
            # several output positions, so those are the axes summed away and
            # the only ones.
            padded = (1,) * (len(grad.shape) - len(shape)) + tuple(shape)
            axes = tuple(
                axis
                for axis, (produced, original) in enumerate(zip(grad.shape, padded))
                if original == 1 and produced != 1
            )
            reduction = Sum(axis=axes, keepdims=True, on_selected_backend=True)
            reduced = (
                apply_operation(reduction, (grad,))
                if is_graph_operand(grad)
                else reduction.forward(grad)
            )
            # Reducing with ``keepdims`` leaves the axes the broadcast added
            # in place; dropping them is a relabelling, not arithmetic.
            gradients.append(
                reduced if reduced.shape == shape else reshape(reduced, shape)
            )
        return gradients


add = Add().forward
