"""Division operation."""

from typing import List, Optional, Union
from tensors.backend import execute_divide, execute_division_denominator_gradient
from tensors.dtype import convert_scalar, resolve_result_dtype, true_division_dtype
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.utils.broadcasting import broadcast_to, broadcast_tensors
from tensors.operations._gradient_shaping import sum_to_shape

Scalar = Union[int, float]


def _negative_product_over_square(
    left: float, right: float, denominator: float
) -> float:
    """Evaluate ``-left * right / denominator**2`` without range loss."""
    return _product_over_denominator_power(
        [-float(left), float(right)], float(denominator), 2
    )


def _product_over_denominator_power(
    factors: list[float], denominator: float, power: int
) -> float:
    """Evaluate a product divided by a denominator power exactly when finite."""
    import math

    denominator = float(denominator)
    if denominator == 0.0:
        raise ZeroDivisionError("Division by zero")
    if any((value == 0.0 for value in factors)):
        return 0.0
    if all((math.isfinite(value) for value in factors + [denominator])):
        numerator = 1
        divisor = 1
        for factor in factors:
            factor_numerator, factor_denominator = factor.as_integer_ratio()
            numerator *= factor_numerator
            divisor *= factor_denominator
        denominator_numerator, denominator_denominator = denominator.as_integer_ratio()
        numerator *= denominator_denominator**power
        divisor *= denominator_numerator**power
        try:
            return numerator / divisor
        except OverflowError:
            return math.inf if numerator * divisor > 0 else -math.inf
    result = 1.0
    for factor in factors:
        result *= factor
    for _ in range(power):
        result /= denominator
    return result


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
    storage = denominator._logical_storage_for(denominator._storage.kind)
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

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        a, b = inputs
        need_numerator, need_denominator = needs_input_grad
        numerator_gradient = (
            sum_to_shape(self.forward(grad, b), a.shape) if need_numerator else None
        )
        if not need_denominator:
            return [numerator_gradient, None]
        expanded_a, expanded_b = broadcast_tensors(a, b)
        storage = execute_division_denominator_gradient(grad, expanded_a, expanded_b)
        denominator_gradient = Tensor._from_owned_storage(
            storage, dtype=grad.dtype, shape=grad.shape
        )
        return [numerator_gradient, sum_to_shape(denominator_gradient, b.shape)]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable VJP for division."""
        left, right = inputs
        need_numerator, need_denominator = needs_input_grad
        from tensors.operations._gradient_shaping import sum_to_shape_graph

        return [
            sum_to_shape_graph(grad / right, left.shape) if need_numerator else None,
            (
                sum_to_shape_graph(
                    _division_denominator_vjp(grad, left, right), right.shape
                )
                if need_denominator
                else None
            ),
        ]


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
        grad, numerator, denominator = _expanded_division_inputs(
            grad, numerator, denominator
        )
        if any((value == 0 for value in denominator._data)):
            raise ZeroDivisionError("Division by zero")
        accelerated = execute_division_denominator_gradient(
            grad, numerator, denominator
        )
        return Tensor._from_owned_storage(
            accelerated, dtype=grad.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        grad, numerator, denominator = inputs
        need_grad, need_numerator, need_denominator = needs_input_grad
        expanded_grad, expanded_numerator, expanded_denominator = (
            _expanded_division_inputs(grad, numerator, denominator)
        )
        expanded_outer = broadcast_to(outer_grad, expanded_grad.shape)
        grad_values = []
        numerator_values = []
        denominator_values = []
        for outer, upstream, value, divisor in zip(
            expanded_outer._data,
            expanded_grad._data,
            expanded_numerator._data,
            expanded_denominator._data,
        ):
            if need_grad:
                grad_values.append(_negative_product_over_square(outer, value, divisor))
            if need_numerator:
                numerator_values.append(
                    _negative_product_over_square(outer, upstream, divisor)
                )
            if not need_denominator:
                continue
            denominator_values.append(
                _product_over_denominator_power(
                    [2.0, float(outer), float(upstream), float(value)],
                    float(divisor),
                    3,
                )
            )
        shape = expanded_grad.shape

        def reduced(values: list[float], target: Tensor) -> Tensor:
            return sum_to_shape(
                Tensor(values, dtype=outer_grad.dtype, shape=shape), target.shape
            )

        return [
            reduced(grad_values, grad) if need_grad else None,
            reduced(numerator_values, numerator) if need_numerator else None,
            reduced(denominator_values, denominator) if need_denominator else None,
        ]

    def backward_graph(self, outer_grad, *inputs, needs_input_grad: tuple[bool, ...]):
        from tensors.operations._gradient_shaping import sum_to_shape_graph

        grad, numerator, denominator = inputs
        need_grad, need_numerator, need_denominator = needs_input_grad
        return [
            (
                sum_to_shape_graph(
                    _division_denominator_vjp(outer_grad, numerator, denominator),
                    grad.shape,
                )
                if need_grad
                else None
            ),
            (
                sum_to_shape_graph(
                    _division_denominator_vjp(outer_grad, grad, denominator),
                    numerator.shape,
                )
                if need_numerator
                else None
            ),
            (
                sum_to_shape_graph(
                    2.0 * outer_grad * grad * numerator / denominator**3.0,
                    denominator.shape,
                )
                if need_denominator
                else None
            ),
        ]


def _division_denominator_vjp(grad, numerator, denominator):
    from tensors.variable import Variable

    operation = DivisionDenominatorGradient()
    return Variable._apply_operation(operation, (grad, numerator, denominator))


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
