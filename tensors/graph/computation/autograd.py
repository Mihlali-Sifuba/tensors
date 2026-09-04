"""The functional reverse-mode differentiation API.

These functions describe *what* differentiation a caller requests. A
:class:`~tensors.graph.computation.Computation` describes *how* that request
executes, so each entry point here resolves the plan owned by an output
Variable and then hands the work to it.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, overload

from ..._typing import TensorLike
from ...tensor import Tensor
from .compiler import validate_outputs
from .computation import Computation
from .gradients import gradient_seed

if TYPE_CHECKING:
    from ...variable import Variable


def computation_for(output: Variable) -> Computation:
    """Return the reusable reverse plan owned by an output Variable.

    The plan is execution state, so it lives on the runtime value rather than
    on a structural graph node. Graph topology is immutable, so later calls
    reuse the pre-resolved plan; each pass allocates its own execution buffers
    and shares none of them. A released plan is replaced rather than reused.
    """
    validate_outputs((output,))
    computation = output._autograd_computation
    if computation is None or computation._released:
        computation = Computation(output)
        output._autograd_computation = computation
    return computation


def backward(
    output: Variable,
    grad: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> None:
    """Differentiate an output through its recorded Computation."""
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")
    computation_for(output).backward(
        grad,
        create_graph=create_graph,
    )


@overload
def grad(
    output: Variable,
    inputs: Variable,
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> Tensor | Variable | None:
    ...


@overload
def grad(
    output: Variable,
    inputs: Iterable[Variable],
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> tuple[Tensor | Variable | None, ...]:
    ...


def grad(
    output: Variable,
    inputs: Variable | Iterable[Variable],
    grad_outputs: TensorLike | None = None,
    *,
    create_graph: bool = False,
) -> Tensor | Variable | None | tuple[Tensor | Variable | None, ...]:
    """Return gradients of ``output`` without modifying any ``.grad`` field.

    Set ``create_graph=True`` when the returned gradients will themselves be
    differentiated.
    """
    from ...variable import Variable

    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")

    single_input = isinstance(inputs, Variable)
    try:
        requested = (inputs,) if single_input else tuple(inputs)
    except TypeError as exc:
        raise TypeError(
            "grad inputs must be a Variable or an iterable of Variables"
        ) from exc
    if not requested:
        raise ValueError("grad requires at least one input Variable")
    for index, variable in enumerate(requested):
        if not isinstance(variable, Variable):
            raise TypeError(
                f"grad input {index} must be a Variable, got "
                f"{type(variable).__name__}"
            )
    computation = computation_for(output)
    computation._validate_recorded_states()
    # Build the reverse demand from the requested inputs so only the paths
    # connecting them to the output are differentiated.
    live = computation._live_slots(requested)
    if create_graph:
        seed = gradient_seed(
            computation.output,
            grad_outputs,
            create_graph=True,
        )
        gradients = computation._backward_graph(seed, live)
    else:
        seed = gradient_seed(computation.output, grad_outputs)
        gradients = computation._backward_values(seed, live)
    result = tuple(gradients.get(variable) for variable in requested)
    return result[0] if single_input else result


__all__ = ["backward", "grad"]
