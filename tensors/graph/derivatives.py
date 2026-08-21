"""Jacobian and Hessian construction from recorded computations."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import contextmanager
from typing import Any

from ..tensor import Tensor
from ..variable import Variable
from .computation import Computation, grad
from .state import isolated_graph_state


def _normalize_inputs(inputs: Any, operation: str) -> tuple[tuple[Variable, ...], bool]:
    """Return validated input Variables and whether one input was supplied."""
    single_input = isinstance(inputs, Variable)
    try:
        requested = (inputs,) if single_input else tuple(inputs)
    except TypeError as exc:
        raise TypeError(
            f"{operation} inputs must be a Variable or an iterable of Variables"
        ) from exc

    if not requested:
        raise ValueError(f"{operation} requires at least one input Variable")
    for index, variable in enumerate(requested):
        if not isinstance(variable, Variable):
            raise TypeError(
                f"{operation} input {index} must be a Variable, got "
                f"{type(variable).__name__}"
            )
        if not variable.requires_grad:
            raise ValueError(
                f"{operation} input {index} must have requires_grad=True"
            )
    return requested, single_input


def _validate_output(output: Any, operation: str) -> Variable:
    """Return a graph-connected Variable suitable for differentiation."""
    if not isinstance(output, Variable) or output.node is None:
        raise TypeError(f"{operation} output must be a Variable with a graph node")
    return output


def _zero_tensor(reference: Variable, shape: tuple[int, ...]) -> Tensor:
    """Return zeros shaped as a derivative and typed like its input."""
    size = 1
    for dimension in shape:
        size *= dimension
    return Tensor([0.0] * size, dtype=reference.dtype, shape=shape)


def _basis(output: Variable, index: int) -> Tensor:
    """Return one row of the identity map over the flattened output."""
    values = [0.0] * output.size
    values[index] = 1.0
    return Tensor(values, dtype=output.dtype, shape=output.shape)


def _saved_gradients(
    output: Variable,
    inputs: Iterable[Variable],
) -> list[tuple[Variable, Any]]:
    """Capture gradients that repeated reverse passes temporarily replace."""
    variables = []
    seen = set()
    for node in Computation(output).nodes:
        variable = node.output_var
        if variable is not None and id(variable) not in seen:
            seen.add(id(variable))
            variables.append(variable)
    for variable in inputs:
        if id(variable) not in seen:
            seen.add(id(variable))
            variables.append(variable)
    return [(variable, variable.grad) for variable in variables]


@contextmanager
def _derivative_graph_scope(
    output: Variable,
    inputs: Iterable[Variable],
    *,
    keep_graph: bool,
):
    """Isolate and release temporary higher-order graph nodes when possible."""
    if keep_graph:
        yield
        return

    source_nodes = list(Computation(output).nodes)
    seen = {id(node) for node in source_nodes}
    for variable in inputs:
        if variable.node is not None and id(variable.node) not in seen:
            seen.add(id(variable.node))
            source_nodes.append(variable.node)
    previous_outputs = [tuple(node._out_edges) for node in source_nodes]

    try:
        with isolated_graph_state():
            yield
    finally:
        for node, edges in zip(source_nodes, previous_outputs):
            node._replace_out_edges(edges)


def _assemble_rows(
    output: Variable,
    input_variable: Variable,
    rows: list[Tensor | Variable],
    *,
    create_graph: bool,
) -> Tensor | Variable:
    """Assemble flattened VJP rows into output-shape-plus-input-shape form."""
    shape = output.shape + input_variable.shape
    if not rows:
        zero = _zero_tensor(input_variable, shape)
        return Variable(zero, requires_grad=False) if create_graph else zero

    if create_graph:
        from ..math import reshape, stack

        return reshape(stack(rows, axis=0), shape)

    values = []
    for row in rows:
        values.extend(row.data._data if isinstance(row, Variable) else row._data)
    return Tensor(values, dtype=rows[0].dtype, shape=shape)


def jacobian(
    output: Variable,
    inputs: Variable | Iterable[Variable],
    *,
    create_graph: bool = False,
) -> Tensor | Variable | tuple[Tensor | Variable, ...]:
    """Return every first derivative of ``output`` with respect to ``inputs``.

    Each result has shape ``output.shape + input.shape``. A single input returns
    one Tensor or Variable; multiple inputs return a tuple in the supplied order.
    Disconnected inputs produce zero Jacobians.
    """
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")
    output = _validate_output(output, "jacobian")
    requested, single_input = _normalize_inputs(inputs, "jacobian")
    rows: list[list[Tensor | Variable]] = [[] for _ in requested]
    saved = _saved_gradients(output, requested)

    try:
        for output_index in range(output.size):
            derivatives = grad(
                output,
                requested,
                grad_outputs=_basis(output, output_index),
                create_graph=create_graph,
            )
            for input_index, (input_variable, derivative) in enumerate(
                zip(requested, derivatives)
            ):
                if derivative is None:
                    zero = _zero_tensor(input_variable, input_variable.shape)
                    derivative = (
                        Variable(zero, requires_grad=False)
                        if create_graph
                        else zero
                    )
                rows[input_index].append(derivative)
    finally:
        for variable, previous_gradient in saved:
            variable.grad = previous_gradient

    results = tuple(
        _assemble_rows(
            output,
            input_variable,
            input_rows,
            create_graph=create_graph,
        )
        for input_variable, input_rows in zip(requested, rows)
    )
    return results[0] if single_input else results


def hessian(
    output: Variable,
    inputs: Variable | Iterable[Variable],
    *,
    create_graph: bool = False,
) -> (
    Tensor
    | Variable
    | tuple[tuple[Tensor | Variable, ...], ...]
):
    """Return the Hessian of a single-element ``output``.

    For one input, the result has shape ``input.shape + input.shape``. For
    multiple inputs, the result is a tuple of tuples of Hessian blocks where
    block ``[i][j]`` has shape ``inputs[i].shape + inputs[j].shape``.
    """
    if not isinstance(create_graph, bool):
        raise TypeError("create_graph must be a bool")
    output = _validate_output(output, "hessian")
    if output.size != 1:
        raise ValueError(
            f"hessian output must contain exactly one element, got {output.size}"
        )
    requested, single_input = _normalize_inputs(inputs, "hessian")
    with _derivative_graph_scope(
        output,
        requested,
        keep_graph=create_graph,
    ):
        first_derivatives = grad(output, requested, create_graph=True)
        blocks = []

        for row_input, first_derivative in zip(requested, first_derivatives):
            if first_derivative is None:
                row = tuple(
                    (
                        Variable(
                            _zero_tensor(
                                column_input,
                                row_input.shape + column_input.shape,
                            ),
                            requires_grad=False,
                        )
                        if create_graph
                        else _zero_tensor(
                            column_input,
                            row_input.shape + column_input.shape,
                        )
                    )
                    for column_input in requested
                )
            else:
                derivative_row = jacobian(
                    first_derivative,
                    requested,
                    create_graph=create_graph,
                )
                row = tuple(derivative_row)
            blocks.append(row)

    result = tuple(blocks)
    return result[0][0] if single_input else result


__all__ = ["hessian", "jacobian"]
