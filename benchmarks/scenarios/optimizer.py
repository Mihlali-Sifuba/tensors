"""Optimizers, split into the phases a training loop pays separately.

A step is not one cost. The first step allocates the optimizer's state, so
it is measured on its own state each sample. Steady steps reuse it.
Gradient preparation validates and possibly converts every gradient before
any parameter is written, and the batched kernels exist precisely so that
many small parameters are not updated one at a time — so the parameter
count is varied independently of the parameter size.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import (
    execute_adam_update,
    execute_adam_updates,
    execute_rmsprop_update,
    execute_sgd_update,
    execute_sgd_updates,
)

from ..case import Case, Group, Unsupported
from ..inputs import ACCELERATED, FLOAT_DTYPES, tensor

#: The optimizers and how to construct one over a parameter list.
OPTIMIZERS: dict[str, Any] = {
    "sgd": lambda parameters: ts.optim.SGD(parameters, learning_rate=0.01),
    "adam": lambda parameters: ts.optim.Adam(parameters, learning_rate=0.001),
    "rmsprop": lambda parameters: ts.optim.RMSprop(parameters, learning_rate=0.01),
}


def _parameters(count: int, size: int, dtype_name: str) -> list[ts.Variable]:
    """Build ``count`` parameters of ``size`` elements each, with gradients."""
    parameters = []
    for _ in range(count):
        parameter = ts.Variable(
            tensor((size,), dtype_name=dtype_name, kind="constant", value=0.5)
        )
        parameter.grad = tensor(
            (size,), dtype_name=dtype_name, kind="constant", value=0.01
        )
        parameters.append(parameter)
    return parameters


def _optimizer_cases(
    backend: str,
    name: str,
    count: int,
    size: int,
    dtype_name: str,
) -> list[Case]:
    """Build first-step, steady-step, and phase cases for one optimizer."""
    total = count * size
    suffix = f"{name}/{dtype_name}/{count}x{size}"
    common: dict[str, Any] = {
        "family": f"optimizer/{name}",
        "dtype": dtype_name,
        "shape": (count, size),
        "elements": total,
        "work_items": total,
        "tags": {
            "curve": f"optimizer-{name}|{dtype_name}|{count}",
            "dtype_pair": f"optimizer-{name}|{count}",
            "optimizer": name,
            "parameter_count": str(count),
        },
    }
    cases: list[Case] = []
    build = OPTIMIZERS[name]

    # -- first step: state allocation included -------------------------
    first_state: dict[str, Any] = {}

    def reset_first() -> None:
        parameters = _parameters(count, size, dtype_name)
        first_state["parameters"] = parameters
        first_state["optimizer"] = build(parameters)

    def run_first() -> None:
        first_state["optimizer"].step()

    first_common = dict(common)
    first_common["tags"] = {
        **common["tags"],
        "curve": f"optimizer-{name}-first|{dtype_name}|{count}",
        "phase": "first-step",
    }
    cases.append(
        Case(
            name=f"optimizer.first_step/{suffix}",
            run=run_first,
            layer="optimizer",
            validate=run_first,
            reset=reset_first,
            single_shot=True,
            description=(
                "the first step on fresh state, which allocates whatever "
                "moments this optimizer keeps"
            ),
            memory=True,
            **first_common,
        )
    )

    # -- steady step: state already allocated --------------------------
    steady_parameters = _parameters(count, size, dtype_name)
    steady_optimizer = build(steady_parameters)
    steady_optimizer.step()

    def reset_steady() -> None:
        # Gradients are consumed by a step only in the sense that the
        # caller normally replaces them; restoring them keeps every sample
        # starting from the same demand.
        for parameter in steady_parameters:
            parameter.grad = tensor(
                (size,), dtype_name=dtype_name, kind="constant", value=0.01
            )

    steady_common = dict(common)
    steady_common["tags"] = {**common["tags"], "phase": "steady-step"}
    cases.append(
        Case(
            name=f"optimizer.steady_step/{suffix}",
            run=steady_optimizer.step,
            layer="optimizer",
            validate=steady_optimizer.step,
            reset=reset_steady,
            description="a step with state already allocated",
            memory=True,
            **steady_common,
        )
    )

    # -- the phases inside a step --------------------------------------
    phase_common = dict(common)
    phase_common["tags"] = {
        **common["tags"],
        "curve": f"optimizer-phase|{dtype_name}|{count}",
    }
    cases.append(
        Case(
            name=f"optimizer.prepared_gradients/{suffix}",
            run=steady_optimizer._prepared_gradients,
            layer="optimizer",
            validate=steady_optimizer._prepared_gradients,
            description=(
                "validate every gradient's shape and dtype before any "
                "parameter is written"
            ),
            **phase_common,
        )
    )

    zero_parameters = _parameters(count, size, dtype_name)
    zero_optimizer = build(zero_parameters)

    def reset_zero() -> None:
        for parameter in zero_parameters:
            parameter.grad = tensor(
                (size,), dtype_name=dtype_name, kind="constant", value=0.01
            )

    cases.append(
        Case(
            name=f"optimizer.zero_grad/{suffix}",
            run=zero_optimizer.zero_grad,
            layer="optimizer",
            validate=zero_optimizer.zero_grad,
            reset=reset_zero,
            description="clear every managed parameter's gradient",
            **phase_common,
        )
    )

    # -- the update kernels, single and batched ------------------------
    if backend in ACCELERATED:
        parameter_tensors = tuple(parameter.data for parameter in steady_parameters)
        gradient_tensors = tuple(
            tensor((size,), dtype_name=dtype_name, kind="constant", value=0.01)
            for _ in range(count)
        )
        kernel_common = dict(common)
        kernel_common["tags"] = {
            **common["tags"],
            "curve": f"optimizer-kernel-{name}|{dtype_name}|{count}",
        }

        # Adam and RMSprop keep their second moment as a scale and a
        # scaled value, so their kernels take both.
        def state() -> ts.Tensor:
            return tensor((size,), dtype_name=dtype_name, kind="constant", value=1.0)

        if name == "sgd":
            single_call = lambda: execute_sgd_update(
                parameter_tensors[0], gradient_tensors[0], 0.01
            )
            batched_call = lambda: execute_sgd_updates(
                parameter_tensors, gradient_tensors, 0.01
            )
        elif name == "adam":
            moment, scale, scaled = state(), state(), state()
            single_call = lambda: execute_adam_update(
                parameter_tensors[0],
                gradient_tensors[0],
                moment,
                scale,
                scaled,
                beta1=0.9,
                beta2=0.999,
                learning_rate=0.001,
                epsilon=1e-8,
                first_correction=0.1,
                second_correction=0.001,
            )
            moments = tuple(state() for _ in range(count))
            scales = tuple(state() for _ in range(count))
            scaled_values = tuple(state() for _ in range(count))
            # The batched entry point takes one bias correction per
            # parameter, since batched states may be at different steps.
            corrections = (0.1,) * count
            second = (0.001,) * count
            batched_call = lambda: execute_adam_updates(
                parameter_tensors,
                gradient_tensors,
                moments,
                scales,
                scaled_values,
                beta1=0.9,
                beta2=0.999,
                learning_rate=0.001,
                epsilon=1e-8,
                first_corrections=corrections,
                second_corrections=second,
            )
        else:
            scale, scaled = state(), state()
            single_call = lambda: execute_rmsprop_update(
                parameter_tensors[0],
                gradient_tensors[0],
                scale,
                scaled,
                rho=0.99,
                learning_rate=0.01,
                epsilon=1e-8,
            )
            batched_call = None

        def validate_single(call: Any = single_call) -> None:
            if call() is None:
                raise Unsupported(
                    "the optimizer kernel declined this configuration and "
                    "deferred to the Python reference implementation"
                )

        single_common = dict(kernel_common)
        single_common["elements"] = size
        single_common["work_items"] = size
        single_common["shape"] = (size,)
        cases.append(
            Case(
                name=f"optimizer.kernel_single/{suffix}",
                run=single_call,
                layer="optimizer",
                validate=validate_single,
                description="one parameter's fused update kernel",
                backends=ACCELERATED,
                **single_common,
            )
        )

        if batched_call is not None and count > 1:

            def validate_batched(call: Any = batched_call) -> None:
                if call() is None:
                    raise Unsupported(
                        "the batched optimizer kernel declined this " "configuration"
                    )

            cases.append(
                Case(
                    name=f"optimizer.kernel_batched/{suffix}",
                    run=batched_call,
                    layer="optimizer",
                    validate=validate_batched,
                    description=(f"one batched kernel updating all {count} parameters"),
                    backends=ACCELERATED,
                    **kernel_common,
                )
            )
    return cases


def groups() -> list[Group]:
    """Return optimizer groups over shape, count, and dtype."""
    result: list[Group] = []

    # (parameter count, elements per parameter) — the two axes separated.
    configurations: tuple[tuple[int, int], ...] = (
        (1, 1),
        (1, 1_000),
        (1, 100_000),
        (1, 1_000_000),
        (4, 1_000),
        (16, 100),
        (64, 100),
        (256, 100),
        (64, 10_000),
        (4, 1_000_000),
    )

    for name in OPTIMIZERS:
        for dtype_name in FLOAT_DTYPES:
            for count, size in configurations:
                total = count * size

                def factory(
                    backend: str,
                    name: str = name,
                    count: int = count,
                    size: int = size,
                    dtype_name: str = dtype_name,
                    total: int = total,
                ) -> Sequence[Case]:
                    if backend == "python" and total > 20_000:
                        raise Unsupported(
                            f"{total} parameter values exceeds the Python "
                            "backend optimizer ceiling"
                        )
                    if backend == "cuda" and total > 4_000_000:
                        raise Unsupported(
                            f"{total} parameter values plus optimizer state "
                            "and temporaries exceeds device memory in this "
                            "run"
                        )
                    return _optimizer_cases(backend, name, count, size, dtype_name)

                result.append(
                    Group(
                        name=(f"optimizer/{name}/{dtype_name}/{count}x{size}"),
                        factory=factory,
                        suite="optimizer",
                    )
                )
    return result


__all__ = ["groups"]
