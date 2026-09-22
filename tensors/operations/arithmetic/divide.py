"""Division operation."""

from typing import List, Optional, Union
from tensors.backend import execute_divide, execute_division_denominator_gradient
from tensors.dtype import convert_scalar, resolve_result_dtype, true_division_dtype
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.broadcasting import broadcast_to
from tensors.operations.vjp import sum_to_shape

Scalar = Union[int, float]


def _denominator_has_zero(denominator: Tensor) -> bool:
    """Whether an integer denominator contains a zero.

    Integer division has no infinity to deliver, so this test is required and
    section 7.2 keeps it. It runs only for integer operands: floating division
    delivers the IEEE result and must never read its denominator, which on a
    device would mean a host synchronisation on every call.

    The test asks the native buffer rather than materialising the tensor, so
    a device denominator costs one synchronisation and no transfer. A view is
    resolved to its logical values first, so elements the denominator does not
    address cannot make it raise.
    """
    storage = denominator._logical_storage_for(denominator.backend_storage.kind)
    buffer = storage.buffer
    if getattr(buffer, "any", None) is None:
        return any(value == 0 for value in buffer)
    return bool((buffer == 0).any())


def _is_integer_division(left: Tensor, right) -> bool:
    """Whether both operands of a division are integers."""
    if left.dtype.kind != "integer":
        return False
    right_dtype = getattr(right, "dtype", None)
    if right_dtype is not None:
        return right_dtype.kind == "integer"
    return True


class Div(Operation):
    """Element-wise division — forward and backward."""

    __slots__ = ()
    name = "div"

    def forward(self, a: Tensor, b: Union[Tensor, Scalar]) -> Tensor:
        """Element-wise division."""
        if not isinstance(b, (int, float, Tensor)):
            raise TypeError(f"Unsupported: {type(b)}")
        # Resolve the declared dtypes first, then let true division adapt
        # the result domain. A scalar converts to the tensor's dtype, which
        # is the conversion target even when the result is floating.
        # Sections 6.2, 6.5 and 7.3.
        if isinstance(b, Tensor):
            other = b
            resolved_dtype = resolve_result_dtype(a.dtype, b.dtype)
        else:
            other = convert_scalar(b, a.dtype)
            resolved_dtype = a.dtype
        dtype = true_division_dtype(resolved_dtype)
        if _is_integer_division(a, b):
            if isinstance(b, Tensor):
                if _denominator_has_zero(b):
                    raise ZeroDivisionError("Division by zero")
            elif other == 0:
                raise ZeroDivisionError("Division by zero")
        if isinstance(b, Tensor):
            shape = a.shape.broadcast_with(b.shape)
            accelerated = execute_divide(
                a, b, dtype=dtype, output_shape=shape
            )
            return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=shape)
        accelerated = execute_divide(
            a, other, dtype=dtype, output_shape=a.shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=a.shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Route the upstream gradient to both operands.

        The numerator's VJP is the upstream gradient divided by the
        denominator, which is division again. The denominator's is the
        range-safe primitive below, applied rather than called, so the
        operands decide what the statement means: given Tensors it
        calculates, and given Variables it records a differentiable graph.
        Both are then reduced over the axes the forward broadcast stretched.

        Both reductions run on the selected backend. The older `sum_to_shape`
        applies a workload threshold and answers a small gradient with Python
        storage, which under an explicit NumPy or CUDA selection is a
        residency violation rather than a performance choice.

        An unrequested operand costs nothing: neither its quotient nor its
        primitive is formed.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        numerator, denominator = inputs
        gradients = []
        for slot, operand in enumerate(inputs):
            if not needs_input_grad[slot]:
                gradients.append(None)
                continue
            if slot == 0:
                contribution = grad / denominator
            else:
                primitive = DivisionDenominatorGradient()
                contribution = (
                    apply_operation(primitive, (grad, numerator, denominator))
                    if is_graph_operand(grad)
                    else primitive.forward(grad, numerator, denominator)
                )
            gradients.append(
                sum_to_shape(contribution, operand.shape)
            )
        return gradients


def _expanded_division_inputs(
    grad: Tensor, numerator: Tensor, denominator: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    shape = grad.shape.broadcast_with(numerator.shape).broadcast_with(denominator.shape)
    return (
        broadcast_to(grad, shape),
        broadcast_to(numerator, shape),
        broadcast_to(denominator, shape),
    )


class DivisionDenominatorGradient(Operation):
    """Differentiable range-safe VJP for a division denominator."""

    __slots__ = ()
    name = "division_denominator_gradient"

    def forward(self, grad: Tensor, numerator: Tensor, denominator: Tensor) -> Tensor:
        """Evaluate ``-grad * numerator / denominator ** 2``.

        A zero denominator is **not** refused here. Section 7.2 gives
        floating division a signed infinity or a NaN, the eager backward
        pass has always delivered that, and the kernels deliver it here
        too. The host scan that used to raise ``ZeroDivisionError`` read
        every denominator value into Python and settled the question once,
        when this operation was recorded, which made it disagree with the
        eager pass it is supposed to reproduce.
        """
        grad, numerator, denominator = _expanded_division_inputs(
            grad, numerator, denominator
        )
        accelerated = execute_division_denominator_gradient(
            grad,
            numerator,
            denominator,
            dtype=grad.dtype,
            output_shape=grad.shape,
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=grad.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Differentiate the denominator VJP on the selected backend.

        This used to iterate over the operands' host values in Python and
        build Python lists, so higher-order division was evaluated by the
        Python reference whatever backend was selected. Each partial is now
        an expression over operations that execute where the selection says.

        Writing ``r = -g * a / b ** 2``, the three partials are

        ``dr/dg = -outer * a / b ** 2``   — this operation, on ``a``
        ``dr/da = -outer * g / b ** 2``   — this operation, on ``g``
        ``dr/db = 2 * outer * g * a / b ** 3``

        The first two reuse the range-safe primitive exactly, so the
        behaviour the range-safe path exists for is preserved where the
        contract names it. The third is factored through that same
        primitive rather than forming ``b ** 3`` — ``2 * o * g * a / b**3``
        is ``-2 * (a / b) * (-o * g / b**2)`` — which keeps the cubed
        denominator from overflowing on its own. This is a deliberate
        change of floating-point behaviour from the exact-rational loop it
        replaced, and it is now the only statement of these partials: the
        separate differentiable path that formed ``b ** 3`` directly is gone.

        The operands are not broadcast here. ``forward`` expands them for the
        kernel that needs it, so doing it again would materialise them
        through the host, and a recorded operand cannot be expanded at all.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        grad, numerator, denominator = inputs
        need_grad, need_numerator, need_denominator = needs_input_grad

        def denominator_vjp(value):
            """This same operation, on ``value`` in the numerator's place."""
            operands = (outer_grad, value, denominator)
            return (
                apply_operation(DivisionDenominatorGradient(), operands)
                if is_graph_operand(outer_grad)
                else DivisionDenominatorGradient().forward(*operands)
            )

        def reduced(value, target):
            return sum_to_shape(value, target.shape)

        grad_partial = reduced(denominator_vjp(numerator), grad) if need_grad else None
        numerator_partial = (
            reduced(denominator_vjp(grad), numerator) if need_numerator else None
        )
        denominator_partial = None
        if need_denominator:
            inner = denominator_vjp(grad)
            denominator_partial = reduced(
                inner * (numerator / denominator) * -2.0, denominator
            )
        return [grad_partial, numerator_partial, denominator_partial]


divide = Div().forward


def divide_scalar(numerator: Scalar, denominator: Tensor) -> Tensor:
    """Return ``numerator / denominator`` for a scalar left operand."""
    converted = convert_scalar(numerator, denominator.dtype)
    dtype = true_division_dtype(denominator.dtype)
    if denominator.dtype.kind == "integer" and _denominator_has_zero(denominator):
        raise ZeroDivisionError("Division by zero")
    numerator = converted
    accelerated = execute_divide(
        numerator, denominator, dtype=dtype, output_shape=denominator.shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=denominator.shape)
