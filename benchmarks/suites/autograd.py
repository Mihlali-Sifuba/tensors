"""Reverse-mode differentiation, and the parts a reverse pass is built from.

Backward is measured against the forward pass it differentiates, over the
same graph, so the ratio between them is a property of the derivative rules
rather than of the workload.

A reverse pass is not only derivative rules, so its components are isolated
too: the mutation-state validation that runs before it, the demand analysis
that decides which VJPs to execute, the accumulation that combines
contributions arriving at a shared value, and the per-operation result
validation. Together with the VJP kernels measured in the reduction, linalg
and convolution suites, that accounts for where reverse time goes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.autograd import computation_for
from tensors.graph.computation.gradients import (
    gradient_seed,
    sum_gradient_values,
    validate_gradients,
)

from ..case import Case, Group, Unsupported
from ..workloads import ceiling_for, FLOAT_DTYPES, GRADIENT_CEILING, tensor


def _paired_cases(
    name: str,
    build: Any,
    *,
    width: int,
    operations: int,
    family: str,
    dtype_name: str = "float64",
    memory: bool = False,
) -> list[Case]:
    """Build matched forward and backward cases over one graph.

    ``build`` returns the output Variable of a freshly traced graph. The
    graph is traced once here; the cases then replay and differentiate it,
    so neither measurement includes tracing.
    """
    output = build()
    computation = Computation(output)
    computation.forward()
    computation.backward()

    common: dict[str, Any] = {
        "family": family,
        "dtype": dtype_name,
        "shape": (width,),
        "elements": width,
        "work_items": operations,
        "tags": {
            "pair": f"autograd-{name}",
            "curve": f"autograd-{name}",
        },
    }
    forward_common = dict(common)
    forward_common["tags"] = {**common["tags"], "phase": "forward"}
    backward_common = dict(common)
    backward_common["tags"] = {**common["tags"], "phase": "backward"}
    return [
        Case(
            name=f"autograd.forward/{name}",
            run=computation.forward,
            layer="graph-replay",
            validate=computation.forward,
            description="forward replay of the graph being differentiated",
            **forward_common,
        ),
        Case(
            name=f"autograd.backward/{name}",
            run=computation.backward,
            layer="autograd",
            validate=computation.backward,
            description="the complete reverse pass over the same graph",
            memory=memory,
            **backward_common,
        ),
    ]


def _single_operation_cases(
    backend: str, operation: str, width: int, dtype_name: str
) -> list[Case]:
    """Differentiate one operation, so the rule is what is measured."""
    left = ts.Variable(
        tensor((width,), dtype_name=dtype_name, kind="ramp")
    )
    right = ts.Variable(
        tensor((width,), dtype_name=dtype_name, kind="constant", value=2.0)
    )

    builders = {
        "add": lambda: left + right,
        "multiply": lambda: left * right,
        "divide": lambda: left / right,
        "power": lambda: left ** right,
        "exp": lambda: ts.exp(left),
        "log": lambda: ts.log(left),
        "tanh": lambda: ts.tanh(left),
        "relu": lambda: ts.relu(left),
        "sum": lambda: ts.sum(left),
        "mean": lambda: ts.mean(left),
        "max": lambda: ts.max(left),
        "std": lambda: ts.std(left),
        "norm": lambda: ts.norm(left),
        "softmax": lambda: ts.sum(ts.softmax(left)),
    }
    if operation not in builders:
        raise Unsupported(f"no builder for operation {operation!r}")

    return _paired_cases(
        f"single-{operation}/{dtype_name}/{width}",
        builders[operation],
        width=width,
        operations=1,
        family=f"autograd/single/{operation}",
        dtype_name=dtype_name,
    )


def _topology_cases(
    backend: str, topology: str, depth: int, width: int
) -> list[Case]:
    """Differentiate graphs whose shape, not size, is what varies."""
    leaf = ts.Variable(
        tensor((width,), dtype_name="float64", kind="constant", value=1.0)
    )

    def chain() -> Any:
        current = leaf
        for _ in range(depth):
            current = current * 1.0009
        return ts.sum(current)

    def branched() -> Any:
        # Several independent paths from one leaf, joined at the end. The
        # reverse pass has to accumulate every path's contribution into
        # that one leaf.
        branches = [leaf * float(index + 1) for index in range(depth)]
        total = branches[0]
        for branch in branches[1:]:
            total = total + branch
        return ts.sum(total)

    def accumulation() -> Any:
        # One shared subexpression consumed repeatedly, so the shared value
        # receives ``depth`` gradient contributions.
        shared = leaf * 2.0
        total = shared
        for _ in range(depth):
            total = total + shared
        return ts.sum(total)

    builders = {
        "chain": chain,
        "branched": branched,
        "accumulation": accumulation,
    }
    if topology not in builders:
        raise Unsupported(f"no builder for topology {topology!r}")

    return _paired_cases(
        f"{topology}/{depth}/{width}",
        builders[topology],
        width=width,
        operations=depth,
        family=f"autograd/{topology}",
        memory=True,
    )


def _component_cases(backend: str, depth: int, width: int) -> list[Case]:
    """Isolate the parts of a reverse pass other than derivative rules."""
    leaf = ts.Variable(
        tensor((width,), dtype_name="float64", kind="constant", value=1.0)
    )
    current = leaf
    for _ in range(depth):
        current = current * 1.0009
    output = ts.sum(current)
    computation = Computation(output)
    computation.forward()
    computation.backward()

    common: dict[str, Any] = {
        "family": "autograd/components",
        "dtype": "float64",
        "shape": (width,),
        "elements": width,
        "work_items": depth,
        "tags": {"curve": f"autograd-components|{width}"},
    }
    cases: list[Case] = []

    cases.append(Case(
        name=f"autograd.validate_states/{depth}/{width}",
        run=computation._validate_recorded_states,
        layer="autograd",
        validate=computation._validate_recorded_states,
        description=(
            "the pre-reverse check that no recorded forward value was "
            "mutated, which walks every instruction and its operands"
        ),
        **common,
    ))
    cases.append(Case(
        name=f"autograd.live_slots_all/{depth}/{width}",
        run=lambda: computation._live_slots(None),
        layer="autograd",
        validate=lambda: computation._live_slots(None),
        description=(
            "demand analysis for a full backward, which asks every slot "
            "whether it requires a gradient"
        ),
        **common,
    ))
    cases.append(Case(
        name=f"autograd.live_slots_selective/{depth}/{width}",
        run=lambda: computation._live_slots((leaf,)),
        layer="autograd",
        validate=lambda: computation._live_slots((leaf,)),
        description=(
            "demand analysis for one requested input, which closes the set "
            "over the paths reaching it"
        ),
        **common,
    ))
    cases.append(Case(
        name=f"autograd.gradient_seed/{depth}/{width}",
        run=lambda: gradient_seed(output, None),
        layer="autograd",
        validate=lambda: gradient_seed(output, None),
        description="construct the upstream gradient a reverse pass starts from",
        **common,
    ))

    # Accumulation, at the fan-in counts a shared value actually sees.
    for terms in (1, 2, 8, 64):
        contributions = [
            tensor((width,), dtype_name="float64", kind="constant", value=1.0)
            for _ in range(terms)
        ]
        accumulate_common = dict(common)
        accumulate_common["work_items"] = terms
        accumulate_common["tags"] = {
            "curve": f"autograd-accumulate|{width}",
        }
        cases.append(Case(
            name=f"autograd.sum_gradient_values/{terms}/{width}",
            run=lambda contributions=contributions: sum_gradient_values(
                list(contributions)
            ),
            layer="autograd",
            validate=lambda contributions=contributions: sum_gradient_values(
                list(contributions)
            ),
            description=(
                f"combine {terms} gradient contributions arriving at one "
                "value"
            ),
            **accumulate_common,
        ))

    # Per-operation result validation, which every rule's output passes
    # through.
    from tensors.ops import Mul

    operation = Mul()
    inputs = (leaf, leaf)
    gradients = (
        tensor((width,), dtype_name="float64", kind="constant", value=1.0),
        tensor((width,), dtype_name="float64", kind="constant", value=1.0),
    )
    cases.append(Case(
        name=f"autograd.validate_gradients/{width}",
        run=lambda: validate_gradients(
            operation, inputs, gradients, (True, True), graph=False
        ),
        layer="autograd",
        validate=lambda: validate_gradients(
            operation, inputs, gradients, (True, True), graph=False
        ),
        description=(
            "check one operation's VJP result against the demand that was "
            "made of it"
        ),
        **common,
    ))
    return cases


def _selective_cases(backend: str, depth: int, width: int) -> list[Case]:
    """Compare a full backward against a pruned, single-input request."""
    leaves = [
        ts.Variable(
            tensor((width,), dtype_name="float64", kind="constant", value=1.0)
        )
        for _ in range(8)
    ]
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    current = total
    for _ in range(depth):
        current = current * 1.0009
    output = ts.sum(current)
    computation_for(output).forward()

    common: dict[str, Any] = {
        "family": "autograd/selective",
        "dtype": "float64",
        "shape": (width,),
        "elements": width,
        "work_items": depth,
        "tags": {"curve": f"autograd-selective|{width}"},
    }
    return [
        Case(
            name=f"autograd.backward_all/{depth}/{width}",
            run=lambda: ts.backward(output),
            layer="autograd",
            validate=lambda: ts.backward(output),
            description=(
                "backward(), which publishes a gradient at every reachable "
                "differentiable Variable"
            ),
            **common,
        ),
        Case(
            name=f"autograd.grad_one_input/{depth}/{width}",
            run=lambda: ts.grad(output, leaves[0]),
            layer="autograd",
            validate=lambda: ts.grad(output, leaves[0]),
            description=(
                "grad() for one of eight leaves, so only the paths reaching "
                "it are differentiated"
            ),
            **common,
        ),
        Case(
            name=f"autograd.grad_all_inputs/{depth}/{width}",
            run=lambda: ts.grad(output, leaves),
            layer="autograd",
            validate=lambda: ts.grad(output, leaves),
            description="grad() for every leaf, requesting all eight",
            **common,
        ),
    ]


def _higher_order_cases(backend: str, width: int) -> list[Case]:
    """Measure differentiable gradients and the derivatives built on them."""
    leaf = ts.Variable(
        tensor((width,), dtype_name="float64", kind="constant", value=1.5)
    )
    common: dict[str, Any] = {
        "family": "autograd/higher-order",
        "dtype": "float64",
        "shape": (width,),
        "elements": width,
        "tags": {"curve": f"autograd-higher-order|{width}"},
        "gc_enabled": True,
    }

    def first_order() -> Any:
        output = ts.sum(leaf * leaf * leaf)
        return ts.grad(output, leaf)

    def second_order() -> Any:
        output = ts.sum(leaf * leaf * leaf)
        first = ts.grad(output, leaf, create_graph=True)
        return ts.grad(ts.sum(first), leaf)

    def jacobian() -> Any:
        output = leaf * leaf
        return ts.jacobian(output, leaf)

    def hessian() -> Any:
        output = ts.sum(leaf * leaf * leaf)
        return ts.hessian(output, leaf)

    def third_order() -> Any:
        output = ts.sum(leaf * leaf * leaf)
        first = ts.grad(output, leaf, create_graph=True)
        second = ts.grad(ts.sum(first), leaf, create_graph=True)
        return ts.grad(ts.sum(second), leaf)

    def gradcheck() -> Any:
        return ts.gradcheck(lambda value: ts.sum(value * value + value * 3.0), leaf)

    cases = [
        Case(
            name=f"autograd.first_order/{width}",
            run=first_order,
            layer="autograd",
            validate=first_order,
            description="trace and differentiate a cubic once",
            **common,
        ),
        Case(
            name=f"autograd.third_order/{width}",
            run=third_order,
            layer="autograd",
            validate=third_order,
            description=(
                "three reverse passes, the first two recorded so the next "
                "can differentiate them"
            ),
            **common,
        ),
        Case(
            name=f"autograd.second_order/{width}",
            run=second_order,
            layer="autograd",
            validate=second_order,
            description=(
                "differentiate a differentiable gradient, which records a "
                "second graph over the first"
            ),
            **common,
        ),
    ]
    # Jacobian and Hessian run one reverse pass per output element, so they
    # are only tractable at small widths.
    if width <= 32:
        cases.append(Case(
            name=f"autograd.jacobian/{width}",
            run=jacobian,
            layer="autograd",
            validate=jacobian,
            description=(
                f"a full Jacobian, which runs {width} reverse passes"
            ),
            **common,
        ))
        cases.append(Case(
            name=f"autograd.gradcheck/{width}",
            run=gradcheck,
            layer="autograd",
            validate=gradcheck,
            description=(
                "reverse-mode gradients verified against finite differences, "
                f"which evaluates the objective {2 * width} more times"
            ),
            **common,
        ))
        cases.append(Case(
            name=f"autograd.hessian/{width}",
            run=hessian,
            layer="autograd",
            validate=hessian,
            description=(
                f"a full Hessian, which runs {width} differentiable reverse "
                "passes and then one more each"
            ),
            **common,
        ))
    return cases


def _matmul_cases(backend: str, side: int) -> list[Case]:
    """Differentiate a matrix product, where the VJP is two more products."""
    left = ts.Variable(
        tensor((side, side), dtype_name="float64", kind="constant", value=0.05)
    )
    right = ts.Variable(
        tensor((side, side), dtype_name="float64", kind="constant", value=0.1)
    )
    return _paired_cases(
        f"matmul/{side}",
        lambda: ts.sum(left @ right),
        width=side * side,
        operations=side ** 3,
        family="autograd/matmul",
        memory=True,
    )


def _broadcast_cases(backend: str, rows: int, columns: int) -> list[Case]:
    """Differentiate a broadcast, which the reverse pass must sum back down."""
    matrix = ts.Variable(
        tensor((rows, columns), dtype_name="float64", kind="constant", value=1.0)
    )
    bias = ts.Variable(
        tensor((columns,), dtype_name="float64", kind="constant", value=0.5)
    )
    return _paired_cases(
        f"broadcast/{rows}x{columns}",
        lambda: ts.sum(matrix + bias),
        width=rows * columns,
        operations=2,
        family="autograd/broadcast",
        memory=True,
    )


def _convolution_cases(backend: str) -> list[Case]:
    """Differentiate a convolution through the public path."""
    inputs = ts.Variable(
        tensor((8, 16, 32, 32), dtype_name="float64", kind="constant", value=0.5)
    )
    kernel = ts.Variable(
        tensor((16, 16, 3, 3), dtype_name="float64", kind="constant", value=0.1)
    )
    return _paired_cases(
        "conv2d/8x16x32x32",
        lambda: ts.sum(ts.conv2d(inputs, kernel, padding=1)),
        width=8 * 16 * 32 * 32,
        operations=8 * 16 * 32 * 32 * 16 * 9,
        family="autograd/conv2d",
        memory=True,
    )


def groups() -> list[Group]:
    """Return autograd groups across rules, topologies, and components."""
    result: list[Group] = []

    single_operations = (
        "add", "multiply", "divide", "power", "exp", "log", "tanh", "relu",
        "sum", "mean", "max", "std", "norm", "softmax",
    )
    for operation in single_operations:
        for dtype_name in FLOAT_DTYPES:
            for width in (1, 1_000, 100_000):
                def single(
                    backend: str,
                    operation: str = operation,
                    width: int = width,
                    dtype_name: str = dtype_name,
                ) -> Sequence[Case]:
                    if width > ceiling_for(backend, GRADIENT_CEILING):
                        raise Unsupported(
                            f"{width} elements exceeds the {backend} "
                            "gradient ceiling"
                        )
                    return _single_operation_cases(
                        backend, operation, width, dtype_name
                    )

                result.append(Group(
                    name=f"autograd/single/{operation}/{dtype_name}/{width}",
                    factory=single,
                    suite="autograd",
                ))

    for topology in ("chain", "branched", "accumulation"):
        for depth in (1, 10, 100, 500):
            for width in (1, 1_000, 100_000):
                def topology_factory(
                    backend: str,
                    topology: str = topology,
                    depth: int = depth,
                    width: int = width,
                ) -> Sequence[Case]:
                    if backend == "python" and depth * width > 20_000:
                        raise Unsupported(
                            "exceeds the Python backend gradient ceiling"
                        )
                    if depth * width > 10_000_000:
                        raise Unsupported(
                            f"a depth-{depth} graph over {width} elements "
                            "would retain more than this run permits"
                        )
                    return _topology_cases(
                        backend, topology, depth, width
                    )

                result.append(Group(
                    name=f"autograd/{topology}/{depth}/{width}",
                    factory=topology_factory,
                    suite="autograd",
                ))

    for depth in (10, 100, 500):
        for width in (1, 1_000):
            def components(
                backend: str, depth: int = depth, width: int = width,
            ) -> Sequence[Case]:
                if backend == "python" and depth * width > 20_000:
                    raise Unsupported(
                        "exceeds the Python backend gradient ceiling"
                    )
                return _component_cases(backend, depth, width)

            result.append(Group(
                name=f"autograd/components/{depth}/{width}",
                factory=components,
                suite="autograd",
            ))

            def selective(
                backend: str, depth: int = depth, width: int = width,
            ) -> Sequence[Case]:
                if backend == "python" and depth * width > 20_000:
                    raise Unsupported(
                        "exceeds the Python backend gradient ceiling"
                    )
                return _selective_cases(backend, depth, width)

            result.append(Group(
                name=f"autograd/selective/{depth}/{width}",
                factory=selective,
                suite="autograd",
            ))

    for width in (1, 8, 32, 1_000):
        def higher_order(backend: str, width: int = width) -> Sequence[Case]:
            if backend == "python" and width > 32:
                raise Unsupported(
                    "higher-order derivatives over this width exceed the "
                    "Python backend ceiling"
                )
            return _higher_order_cases(backend, width)

        result.append(Group(
            name=f"autograd/higher-order/{width}",
            factory=higher_order,
            suite="autograd",
        ))

    for side in (8, 64, 256, 512):
        def matmul(backend: str, side: int = side) -> Sequence[Case]:
            if backend == "python" and side > 32:
                raise Unsupported(
                    "exceeds the Python backend matrix-gradient ceiling"
                )
            return _matmul_cases(backend, side)

        result.append(Group(
            name=f"autograd/matmul/{side}",
            factory=matmul,
            suite="autograd",
        ))

    for rows, columns in ((8, 64), (256, 256), (1_024, 512)):
        def broadcast(
            backend: str, rows: int = rows, columns: int = columns,
        ) -> Sequence[Case]:
            if backend == "python" and rows * columns > 20_000:
                raise Unsupported(
                    "exceeds the Python backend gradient ceiling"
                )
            return _broadcast_cases(backend, rows, columns)

        result.append(Group(
            name=f"autograd/broadcast/{rows}x{columns}",
            factory=broadcast,
            suite="autograd",
        ))

    def convolution(backend: str) -> Sequence[Case]:
        if backend == "python":
            raise Unsupported(
                "differentiating this convolution through the reference "
                "implementation would dominate the run"
            )
        return _convolution_cases(backend)

    result.append(Group(
        name="autograd/conv2d",
        factory=convolution,
        suite="autograd",
    ))
    return result


__all__ = ["groups"]
