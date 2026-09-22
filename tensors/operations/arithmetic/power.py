"""Element-wise exponentiation and its differentiation rules."""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union, overload
from tensors.backend import (
    execute_power,
    execute_power_base_base_gradient,
    execute_power_base_gradient,
    execute_power_exponent_exponent_gradient,
    execute_power_exponent_gradient,
    execute_power_mixed_gradient,
)
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.dtype import DataType, resolve_power, resolve_power_scalar_base
from tensors.operations.base import Operation
from tensors.tensor import Tensor

if TYPE_CHECKING:
    from tensors.variable import Variable
from tensors.operations.gradient_primitives import sum_to_shape

Scalar = Union[int, float]


class Pow(Operation):
    """Element-wise exponentiation with reverse-mode gradient rules."""

    __slots__ = ()
    name = "pow"

    def forward(self, base: Tensor, exponent: Tensor | Scalar) -> Tensor:
        """Raise every element in ``base`` to ``exponent``."""
        if not isinstance(exponent, (int, float, Tensor)):
            raise TypeError(f"Unsupported exponent type: {type(exponent)}")
        # Promotion for a typed exponent, conversion for a scalar; section
        # 12.5. No element value is read.
        dtype, exponent = resolve_power(base.dtype, exponent)
        if dtype.kind == "integer":
            # Section 12.4.2, applied once the result dtype is settled.
            # The dtype is decided first, from declarations alone, and only
            # then is the exponent's domain examined: a negative exponent
            # never promotes the result to a floating dtype, it refuses the
            # operation. Integer exponentiation has no fractional value to
            # deliver, and a floating power delivers the IEEE result and
            # must never read its exponent, which is why this is reached
            # only for an integer result.
            if isinstance(exponent, Tensor):
                # The native buffer answers this, so a device exponent costs
                # one reduction and one scalar transfer rather than a copy of
                # the tensor. A view is resolved to its logical values first,
                # so an element the exponent does not address cannot make it
                # raise.
                buffer = exponent._logical_storage_for(
                    exponent.backend_storage.kind
                ).buffer
                negative = (
                    any(value < 0 for value in buffer)
                    if getattr(buffer, "any", None) is None
                    else bool((buffer < 0).any())
                )
            else:
                negative = exponent < 0
            if negative:
                raise ValueError(
                    "integer exponentiation requires a non-negative exponent; "
                    + dtype.name
                    + " cannot represent a reciprocal. Cast the base to a floating "
                    "dtype, for example base.astype(ts.float64) ** exponent"
                )
        output_shape = (
            base.shape.broadcast_with(exponent.shape)
            if isinstance(exponent, Tensor)
            else base.shape
        )
        accelerated = execute_power(
            base, exponent, dtype=dtype, output_shape=output_shape
        )
        return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=output_shape)

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Return the requested VJPs for a power invocation.

        Each gradient is the range-safe primitive of section 12.7 applied to
        the upstream gradient and both operands, reduced afterwards over the
        axes the forward broadcast stretched. The primitive is applied as an
        operation rather than called as a function, so the operands decide
        what the statement means: given Tensors it calculates, and given
        Variables it records a differentiable graph. Rule G5 lives inside it,
        so each gradient carries its own operand's declared dtype rather than
        the upstream gradient's.

        No operand is inspected here. The exponent-gradient domain checks
        that used to stand in this place read both tensors to the host,
        raised for a negative or zero base, and in raising discarded the
        *base* gradient as well — the case section 12.7.3 works through.
        Rules G1 to G3 replace all three behaviours: each requested gradient
        is computed on its own, an absent derivative is NaN rather than an
        exception, and no host synchronisation detects any of it.

        Only a requested derivative is calculated, and a derivative's domain
        check runs only when the derivative it guards was requested.
        """
        from tensors.graph.expression import apply_operation, is_graph_operand

        gradients = []
        for operation, operand, requested in (
            (PowerBaseVJP(), inputs[0], needs_input_grad[0]),
            (PowerExponentVJP(), inputs[1], needs_input_grad[1]),
        ):
            if not requested:
                gradients.append(None)
                continue
            contribution = (
                apply_operation(operation, (grad, inputs[0], inputs[1]))
                if is_graph_operand(grad)
                else operation.forward(grad, inputs[0], inputs[1])
            )
            gradients.append(sum_to_shape(contribution, operand.shape))
        return gradients


def _second_partial_shape(outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor):
    """The shape a second partial is calculated at, before it is reduced."""
    return (
        outer.shape.broadcast_with(grad.shape)
        .broadcast_with(base.shape)
        .broadcast_with(exponent.shape)
    )


def _no_third_derivative(operation: Operation) -> NotImplementedError:
    """Report the order this rule stops at, rather than failing obscurely."""
    return NotImplementedError(
        f"{operation.name} has no derivative rule, so a power cannot be "
        "differentiated a third time. Its own derivative would be one of the "
        "third partials of a power, and none of those is implemented; second "
        "derivatives are unaffected."
    )


class PowerBaseBaseVJP(Operation):
    """Second partial of a power by its base, weighted by both gradients."""

    __slots__ = ()
    name = "power_base_base_vjp"

    def forward(
        self, outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
    ) -> Tensor:
        """Apply ``exponent * (exponent - 1) * base ** (exponent - 2)``.

        The operands are not broadcast here: each backend's kernel broadcasts
        them natively, and expanding first would materialise them through the
        host, which is the read a gradient is specified not to perform.
        """
        accelerated = execute_power_base_base_gradient(outer, grad, base, exponent)
        # Rule G5: the base's declared dtype, not the upstream gradient's.
        return Tensor._from_owned_storage(
            accelerated,
            dtype=base.dtype,
            shape=_second_partial_shape(outer, grad, base, exponent),
        )

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        raise _no_third_derivative(self)


class PowerMixedVJP(Operation):
    """Mixed second partial of a power, weighted by both gradients.

    Differentiating the base VJP by the exponent and the exponent VJP by the
    base give the same mixed partial, so both reach this one operation rather
    than each carrying a statement of it that could drift from the other.

    It is configured with a dtype because that one partial is two different
    gradients: the exponent's when the base VJP is differentiated and the
    base's when the exponent VJP is. Rule G5 gives each the dtype of the
    operand it belongs to, and only the caller knows which it is asking for.
    """

    __slots__ = ("dtype",)
    name = "power_mixed_vjp"

    def __init__(self, *, dtype: DataType) -> None:
        object.__setattr__(self, "dtype", dtype)

    def forward(
        self, outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
    ) -> Tensor:
        """Apply ``base ** (exponent - 1) * (1 + exponent * log(base))``."""
        accelerated = execute_power_mixed_gradient(
            outer, grad, base, exponent, dtype=self.dtype
        )
        return Tensor._from_owned_storage(
            accelerated,
            dtype=self.dtype,
            shape=_second_partial_shape(outer, grad, base, exponent),
        )

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        raise _no_third_derivative(self)


class PowerExponentExponentVJP(Operation):
    """Second partial of a power by its exponent, weighted by both gradients."""

    __slots__ = ()
    name = "power_exponent_exponent_vjp"

    def forward(
        self, outer: Tensor, grad: Tensor, base: Tensor, exponent: Tensor
    ) -> Tensor:
        """Apply ``base ** exponent * log(base) ** 2``."""
        accelerated = execute_power_exponent_exponent_gradient(
            outer, grad, base, exponent
        )
        # Rule G5: the exponent's declared dtype.
        return Tensor._from_owned_storage(
            accelerated,
            dtype=exponent.dtype,
            shape=_second_partial_shape(outer, grad, base, exponent),
        )

    def backward(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        raise _no_third_derivative(self)


def _second_partials(
    operation: Operation,
    outer: Tensor,
    grad: Tensor,
    base: Tensor,
    exponent: Tensor,
    needs_input_grad: tuple[bool, ...],
    base_rule: Operation,
    exponent_rule: Operation,
) -> List[Optional[Tensor]]:
    """Differentiate one of power's VJPs, by the rules its caller names.

    The two VJPs differ only in which second partial belongs to which
    operand: the base VJP is differentiated by the base into ∂²f/∂b² and by
    the exponent into the mixed partial, and the exponent VJP the other way
    round. Everything else — that each VJP is linear in the upstream gradient
    so its derivative by it is itself, which partial is skipped when it is not
    requested, and the reduction back to each operand's shape — is the same
    statement twice, so it is written once and the caller passes the two rules
    that differ.

    Each partial is applied as an operation, so the operands decide what the
    statements mean and the whole of a reverse-over-reverse pass stays on the
    selected backend. Nothing here reads an element.
    """
    from tensors.graph.expression import apply_operation, is_graph_operand

    def apply(rule: Operation, operands: tuple[Tensor, ...]) -> Tensor:
        return (
            apply_operation(rule, operands)
            if is_graph_operand(operands[0])
            else rule.forward(*operands)
        )

    need_grad, need_base, need_exponent = needs_input_grad
    return [
        (
            # This VJP is linear in the upstream gradient, so its derivative
            # by it is the VJP itself, evaluated at the outer gradient.
            sum_to_shape(apply(operation, (outer, base, exponent)), grad.shape)
            if need_grad
            else None
        ),
        (
            sum_to_shape(apply(base_rule, (outer, grad, base, exponent)), base.shape)
            if need_base
            else None
        ),
        (
            sum_to_shape(
                apply(exponent_rule, (outer, grad, base, exponent)), exponent.shape
            )
            if need_exponent
            else None
        ),
    ]


class PowerBaseVJP(Operation):
    """Differentiable range-safe VJP with respect to a power base."""

    __slots__ = ()
    name = "power_base_vjp"

    def forward(self, grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        # The operands are not broadcast here. Each backend's kernel already
        # broadcasts them natively, and expanding first materialised them
        # through the host, which is the read this gradient is specified not
        # to perform.
        accelerated = execute_power_base_gradient(grad, base, exponent)
        # Rule G5: the base's declared dtype, not the upstream gradient's.
        return Tensor._from_owned_storage(
            accelerated, dtype=base.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Differentiate ``upstream * exponent * base ** (exponent - 1)``.

        Its derivative by the base is ∂²f/∂b² and by the exponent is the
        mixed partial; both are backend primitives, because each is a few
        small factors times a power whose range the grouping has to protect,
        and neither can be assembled from operations that stay on the
        selected backend.

        This replaced a host loop that read every operand back, raised at a
        zero and at a negative base, and in raising discarded the partials
        that did exist alongside the one that did not. An absent derivative
        is now NaN, as rules G1 to G3 require of a power's derivatives.
        """
        grad, base, exponent = inputs
        return _second_partials(
            self,
            outer_grad,
            grad,
            base,
            exponent,
            needs_input_grad,
            PowerBaseBaseVJP(),
            PowerMixedVJP(dtype=exponent.dtype),
        )


class PowerExponentVJP(Operation):
    """Differentiable range-safe VJP with respect to a power exponent."""

    __slots__ = ()
    name = "power_exponent_vjp"

    def forward(self, grad: Tensor, base: Tensor, exponent: Tensor) -> Tensor:
        # The operands are not broadcast here. Each backend's kernel already
        # broadcasts them natively, and expanding first materialised them
        # through the host, which is the read this gradient is specified not
        # to perform.
        accelerated = execute_power_exponent_gradient(grad, base, exponent)
        # Rule G5: the exponent's declared dtype.
        return Tensor._from_owned_storage(
            accelerated, dtype=exponent.dtype, shape=grad.shape
        )

    def backward(
        self, outer_grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        """Differentiate ``upstream * base ** exponent * log(base)``.

        Its derivative by the base is the mixed partial — the same operation
        the base VJP reaches, carrying the base's dtype here instead of the
        exponent's — and by the exponent is ∂²f/∂e².
        """
        grad, base, exponent = inputs
        return _second_partials(
            self,
            outer_grad,
            grad,
            base,
            exponent,
            needs_input_grad,
            PowerMixedVJP(dtype=base.dtype),
            PowerExponentExponentVJP(),
        )


@overload
def pow(base: Variable, exponent: TensorLike) -> Variable: ...


@overload
def pow(base: TensorLike, exponent: Variable) -> Variable: ...


@overload
def pow(base: TensorData, exponent: TensorData) -> Tensor: ...


def pow(base: TensorLike, exponent: TensorLike) -> TensorResult:
    """Return the element-wise power of two Tensors, Variables, or scalars."""
    return base**exponent


power = Pow().forward


def power_scalar_base(base: Scalar, exponent: Tensor) -> Tensor:
    """Return ``base`` raised element-wise to ``exponent`` for a scalar base.

    The scalar base converts under rule S-p (section 12.5.3), which reads no
    element of ``exponent``.
    """
    dtype, base = resolve_power_scalar_base(base, exponent.dtype)
    if dtype.kind == "integer":
        # Section 12.4.2, as in Pow.forward: the dtype is settled from
        # declarations alone and only then is the exponent's domain examined.
        # The native buffer answers it, so a device exponent costs one
        # reduction and one scalar transfer rather than a copy of the tensor.
        buffer = exponent._logical_storage_for(exponent.backend_storage.kind).buffer
        negative = (
            any(value < 0 for value in buffer)
            if getattr(buffer, "any", None) is None
            else bool((buffer < 0).any())
        )
        if negative:
            raise ValueError(
                "integer exponentiation requires a non-negative exponent; "
                + dtype.name
                + " cannot represent a reciprocal. Cast the base to a floating "
                "dtype, for example base.astype(ts.float64) ** exponent"
            )
    accelerated = execute_power(
        base, exponent, dtype=dtype, output_shape=exponent.shape
    )
    return Tensor._from_owned_storage(accelerated, dtype=dtype, shape=exponent.shape)


__all__ = ["Pow", "pow", "power", "power_scalar_base"]
