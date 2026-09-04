"""The generic gradient mechanics a reverse pass is built from.

A :class:`~tensors.graph.computation.computation.Computation` decides which
derivatives a reverse pass needs and in what order to execute them. These
functions are what it uses along the way: they construct the upstream seed,
combine the contributions arriving at one Variable, and check what an
operation returned against what was asked of it. None of them knows about
instructions, slots, or execution order, so fused and ordinary reverse
execution can share them.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..._typing import TensorLike
from ...tensor import Tensor

if TYPE_CHECKING:
    from ...ops.operation import Operation
    from ...variable import Variable


def gradient_seed(
    output: Variable,
    grad: TensorLike | None,
    *,
    create_graph: bool = False,
) -> Any:
    """Return a validated upstream gradient, optionally as a Variable."""
    from ...variable import Variable

    if grad is None:
        from ...creation import ones

        typecode = (
            output.dtype.typecode
            if output.dtype.typecode in {"f", "d"}
            else "d"
        )
        seed = ones(
            output.data.shape,
            dtype=typecode,
        )
    elif isinstance(grad, Variable):
        seed = grad
    elif isinstance(grad, (int, float)) and output.data.shape == ():
        seed = Tensor([grad], shape=())
    else:
        seed = grad if isinstance(grad, Tensor) else Tensor(grad)

    seed_shape = seed.shape if isinstance(seed, Variable) else seed.shape
    if seed_shape != output.data.shape:
        raise ValueError(
            f"Gradient shape {seed_shape} does not match output shape "
            f"{output.data.shape}"
        )

    output_dtype = (
        output.dtype
        if output.dtype.typecode in {"f", "d"}
        else None
    )
    if output_dtype is not None and seed.dtype != output_dtype:
        if isinstance(seed, Variable):
            if create_graph and seed.requires_grad:
                raise TypeError(
                    "A differentiable gradient seed must have the same "
                    "dtype as the output"
                )
            seed = Variable(
                seed.data.astype(output_dtype),
                requires_grad=False,
            )
        else:
            seed = seed.astype(output_dtype)

    if create_graph:
        return seed if isinstance(seed, Variable) else Variable(seed, requires_grad=False)
    return seed.data if isinstance(seed, Variable) else seed


def sum_gradient_values(gradients: list[Tensor]) -> Tensor:
    """Combine gradient contributions without order-dependent overflow."""
    if len(gradients) == 1:
        return gradients[0]

    from ...backend import get_backend

    first = gradients[0]
    if get_backend() != "python" and first.size >= 32:
        from ...math import stack
        from ...math import sum as tensor_sum

        return tensor_sum(stack(gradients, axis=0), axis=0)

    from ...math.sum import _stable_float_sum

    values = [
        _stable_float_sum([
            float(gradient._data[index]) for gradient in gradients
        ])
        for index in range(first.size)
    ]
    return Tensor(values, dtype=first.dtype, shape=first.shape)


def sum_gradient_graph(gradients: list[Variable]) -> Variable:
    """Differentiably combine gradient contributions with stable summation."""
    if len(gradients) == 1:
        return gradients[0]

    from ...math import stack, sum

    return sum(stack(gradients, axis=0), axis=0)


def validate_gradients(
    operation: Operation,
    inputs: Sequence[Variable],
    gradients: Any,
    needs_input_grad: tuple[bool, ...],
    *,
    graph: bool,
) -> tuple[Any, ...]:
    """Validate an operation's VJP result against the requested demand.

    A requested input must receive a value of the expected type, shape,
    and dtype. An unrequested input must receive ``None``: returning a
    value there means the operation calculated work the reverse pass did
    not ask for, which makes the demand contract enforceable rather than
    advisory. A real zero remains a valid answer for a requested
    derivative whose value is mathematically zero.
    """
    from ...variable import Variable

    label = operation.name
    mode = "backward_graph" if graph else "backward"
    try:
        results = tuple(gradients)
    except TypeError as exc:
        raise TypeError(
            f"{label} backward must return one gradient per input"
        ) from exc

    if len(results) != len(inputs):
        raise RuntimeError(
            f"{label} backward returned {len(results)} gradients for "
            f"{len(inputs)} inputs"
        )

    expected_type = Variable if graph else Tensor
    validated = list(results)
    for index, (input_variable, gradient, wanted) in enumerate(
        zip(inputs, results, needs_input_grad)
    ):
        if not wanted:
            if gradient is not None:
                raise RuntimeError(
                    f"{label} {mode} returned a gradient for input "
                    f"{index}, which this reverse pass did not request; "
                    "return None for an unrequested derivative"
                )
            continue
        if gradient is None:
            raise RuntimeError(
                f"{label} {mode} returned None for input {index}, whose "
                "gradient this reverse pass requested"
            )
        if not isinstance(gradient, expected_type):
            raise TypeError(
                f"{label} {mode} gradient {index} must be a "
                f"{expected_type.__name__}, got {type(gradient).__name__}"
            )
        if gradient.shape != input_variable.shape:
            raise ValueError(
                f"{label} backward gradient {index} has shape "
                f"{gradient.shape}; expected {input_variable.shape}"
            )
        if (
            input_variable.requires_grad
            and gradient.dtype != input_variable.dtype
        ):
            validated[index] = gradient.astype(input_variable.dtype)
    return tuple(validated)
