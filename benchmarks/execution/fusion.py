"""Fused execution against every alternative way to run the same chain.

Fusion turns a run of elementwise instructions into one kernel, so the
comparison it has to win is not against itself but against the four other
ways the same expression can be evaluated:

``public``
    plain Tensor operations, one kernel and one temporary per step.
``variable``
    the same chain through eager Variables, which also records a graph.
``graph-replay``
    the compiled program with fusion declined, so each instruction runs.
``fusion``
    the compiled program with its fusion plan.
``provider``
    the same arithmetic written directly against NumPy or CuPy.

Depth and width are swept independently, because fusion saves temporaries
(a function of width) by paying a plan and a launch (a function of depth).
The crossover is where those meet.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.graph import Computation
from tensors.graph.computation.fusion import plan_fusions

from ..case import Case, Group, Unsupported
from ..inputs import (
    ACCELERATED,
    FLOAT_DTYPES,
    provider_array,
    provider_module,
    tensor,
)

#: Chain depths, from a single operation to a long expression.
DEPTHS: tuple[int, ...] = (1, 2, 5, 10, 25, 50, 100)

#: Tensor widths, from scalar-like to millions of elements.
WIDTHS: tuple[int, ...] = (1, 100, 10_000, 1_000_000)


def _chain_expression(value: Any, depth: int) -> Any:
    """Apply a fusible elementwise chain of ``depth`` steps.

    The steps alternate between a binary operation with a literal and a
    unary transform, which is what a fusible run looks like in practice and
    keeps every intermediate finite.
    """
    current = value
    for index in range(depth):
        step = index % 4
        if step == 0:
            current = current * 1.0009
        elif step == 1:
            current = current + 0.0001
        elif step == 2:
            current = ts.tanh(current)
        else:
            current = current - 0.0001
    return current


def _fusion_cases(backend: str, depth: int, width: int, dtype_name: str) -> list[Case]:
    """Build every way of evaluating one chain at one depth and width."""
    suffix = f"{dtype_name}/d{depth}/w{width}"
    common: dict[str, Any] = {
        "family": "fusion/chain",
        "dtype": dtype_name,
        "shape": (width,),
        "elements": width,
        "work_items": depth * width,
        "tags": {
            "ladder": f"fusion-chain|{suffix}",
            "curve": f"fusion-chain|{dtype_name}|d{depth}",
            "dtype_pair": f"fusion-chain|d{depth}|w{width}",
            "pair": f"fusion-chain|{suffix}",
            "depth": str(depth),
        },
    }
    cases: list[Case] = []
    value = tensor((width,), dtype_name=dtype_name, kind="constant", value=0.5)

    # -- raw provider ---------------------------------------------------
    if backend in ACCELERATED:
        provider = provider_module(backend)
        raw = provider_array(
            provider,
            (width,),
            dtype_name=dtype_name,
            kind="constant",
            value=0.5,
        )

        def run_provider() -> Any:
            current = raw
            for index in range(depth):
                step = index % 4
                if step == 0:
                    current = current * 1.0009
                elif step == 1:
                    current = current + 0.0001
                elif step == 2:
                    current = provider.tanh(current)
                else:
                    current = current - 0.0001
            return current

        cases.append(
            Case(
                name=f"provider.chain/{suffix}",
                run=run_provider,
                layer="provider",
                validate=run_provider,
                description=(
                    "the same chain written directly against the provider, one "
                    "kernel and one temporary per step"
                ),
                backends=ACCELERATED,
                **common,
            )
        )

    # -- plain Tensor ---------------------------------------------------
    cases.append(
        Case(
            name=f"public.chain/{suffix}",
            run=lambda: _chain_expression(value, depth),
            layer="public",
            validate=lambda: _chain_expression(value, depth),
            description="the chain as plain Tensor operations",
            memory=True,
            **common,
        )
    )

    # -- eager Variable -------------------------------------------------
    variable = ts.Variable(value, requires_grad=False)
    cases.append(
        Case(
            name=f"variable.chain/{suffix}",
            run=lambda: _chain_expression(variable, depth),
            layer="variable",
            validate=lambda: _chain_expression(variable, depth),
            description="the chain through eager Variables",
            gc_enabled=True,
            **common,
        )
    )

    # -- compiled replay, with and without its fusion plan --------------
    traced = _chain_expression(variable, depth)
    computation = Computation(traced)
    computation.forward()
    fusion_plan = computation._fusion.resolve()
    fused_runs = len(fusion_plan[0])
    instruction_count = len(computation._instructions)

    fusion_common = dict(common)
    fusion_common["tags"] = {
        **common["tags"],
        "fused_runs": str(fused_runs),
        "instructions": str(instruction_count),
    }
    cases.append(
        Case(
            name=f"fusion.replay/{suffix}",
            run=computation.forward,
            layer="fusion",
            validate=computation.forward,
            description=(
                f"compiled replay with its fusion plan: {fused_runs} fused "
                f"run(s) over {instruction_count} instructions"
            ),
            memory=True,
            **fusion_common,
        )
    )

    # The same program with fusion declined. A Computation resolves its plan
    # through a holder, so replacing the resolved plan with an empty one
    # makes the identical instruction sequence run unfused.
    unfused = Computation(traced)
    unfused.forward()
    unfused._fusion._plan = ({}, {})
    unfused_common = dict(common)
    unfused_common["tags"] = {
        **common["tags"],
        "curve": f"fusion-unfused|{dtype_name}|d{depth}",
    }
    cases.append(
        Case(
            name=f"fusion.replay_unfused/{suffix}",
            run=unfused.forward,
            layer="graph-replay",
            validate=unfused.forward,
            description=(
                "the identical compiled program with its fusion plan emptied, "
                "so every instruction runs on its own"
            ),
            **unfused_common,
        )
    )

    # -- planning cost --------------------------------------------------
    variables = tuple(node.variable for node in computation._variable_nodes)
    instructions = computation._instructions
    plan_common = dict(common)
    plan_common["work_items"] = instruction_count
    plan_common["tags"] = {
        **common["tags"],
        "curve": f"fusion-planning|d{depth}",
    }
    cases.append(
        Case(
            name=f"fusion.plan/{suffix}",
            run=lambda: plan_fusions(instructions, variables),
            layer="fusion",
            validate=lambda: plan_fusions(instructions, variables),
            description=(
                "recognizing the fusible runs in an instruction sequence, which "
                "a Computation does once per program"
            ),
            **plan_common,
        )
    )

    # -- the fused reverse pass ----------------------------------------
    differentiable = ts.Variable(value, requires_grad=True)
    reverse_output = _chain_expression(differentiable, depth)
    reverse = Computation(reverse_output)
    reverse.forward()
    reverse.backward()
    backward_common = dict(common)
    backward_common["tags"] = {
        **common["tags"],
        "phase": "backward",
        "curve": f"fusion-backward|{dtype_name}|d{depth}",
        "pair": f"fusion-reverse|{suffix}",
    }
    cases.append(
        Case(
            name=f"fusion.backward/{suffix}",
            run=reverse.backward,
            layer="autograd",
            validate=reverse.backward,
            description="the reverse pass over the fused chain",
            memory=True,
            **backward_common,
        )
    )
    forward_pair = dict(common)
    forward_pair["tags"] = {
        **common["tags"],
        "phase": "forward",
        "pair": f"fusion-reverse|{suffix}",
    }
    cases.append(
        Case(
            name=f"fusion.forward_paired/{suffix}",
            run=reverse.forward,
            layer="fusion",
            validate=reverse.forward,
            description="the forward replay its reverse pass is compared against",
            **forward_pair,
        )
    )
    return cases


def _operand_cases(backend: str, operands: int, width: int) -> list[Case]:
    """Vary how many distinct tensors a fused chain reads.

    A fused kernel has to address every external source it reads, so the
    operand count is a separate axis from the chain depth.
    """
    sources = [
        ts.Variable(
            tensor(
                (width,),
                dtype_name="float64",
                kind="constant",
                value=0.5 + index / 100.0,
            ),
            requires_grad=False,
        )
        for index in range(operands)
    ]
    common: dict[str, Any] = {
        "family": "fusion/operands",
        "dtype": "float64",
        "shape": (width,),
        "elements": width,
        "work_items": operands * width,
        "tags": {
            "curve": f"fusion-operands|w{width}",
            "operands": str(operands),
        },
    }

    def build() -> Any:
        current = sources[0]
        for source in sources[1:]:
            current = current + source
        return current

    traced = build()
    computation = Computation(traced)
    computation.forward()
    fused_runs = len(computation._fusion.resolve()[0])

    tensors = [source.data for source in sources]

    def run_public() -> Any:
        current = tensors[0]
        for item in tensors[1:]:
            current = current + item
        return current

    return [
        Case(
            name=f"public.operands/{operands}/{width}",
            run=run_public,
            layer="public",
            validate=run_public,
            description=f"summing {operands} tensors as plain Tensors",
            **common,
        ),
        Case(
            name=f"fusion.operands/{operands}/{width}",
            run=computation.forward,
            layer="fusion",
            validate=computation.forward,
            description=(
                f"summing {operands} tensors through a compiled program "
                f"with {fused_runs} fused run(s)"
            ),
            **common,
        ),
    ]


def groups() -> list[Group]:
    """Return fusion groups over depth, width, dtype, and operand count."""
    result: list[Group] = []

    for dtype_name in FLOAT_DTYPES:
        for depth in DEPTHS:
            for width in WIDTHS:

                def factory(
                    backend: str,
                    depth: int = depth,
                    width: int = width,
                    dtype_name: str = dtype_name,
                ) -> Sequence[Case]:
                    if backend == "python" and depth * width > 200_000:
                        raise Unsupported(
                            f"a depth-{depth} chain over {width} elements "
                            "exceeds the Python backend fusion ceiling"
                        )
                    # The point of the comparison is that the unfused
                    # alternatives hold one temporary per step, so the
                    # ceiling is set by them: depth x width x 8 bytes, three
                    # times over for the provider, Tensor and Variable
                    # chains alive in one group.
                    if depth * width > 20_000_000:
                        raise Unsupported(
                            f"a depth-{depth} chain over {width} elements "
                            f"would hold about "
                            f"{depth * width * 8 / 1e9:.2f} GB of unfused "
                            "temporaries per alternative, above what this "
                            "run permits; the crossover is established by "
                            "the smaller widths"
                        )
                    return _fusion_cases(backend, depth, width, dtype_name)

                result.append(
                    Group(
                        name=f"fusion/chain/{dtype_name}/d{depth}/w{width}",
                        factory=factory,
                        suite="fusion",
                    )
                )

    for operands in (2, 4, 8, 16):
        for width in (100, 10_000, 1_000_000):

            def operand_factory(
                backend: str,
                operands: int = operands,
                width: int = width,
            ) -> Sequence[Case]:
                if backend == "python" and operands * width > 200_000:
                    raise Unsupported("exceeds the Python backend fusion ceiling")
                return _operand_cases(backend, operands, width)

            result.append(
                Group(
                    name=f"fusion/operands/{operands}/{width}",
                    factory=operand_factory,
                    suite="fusion",
                )
            )
    return result


__all__ = ["groups"]
