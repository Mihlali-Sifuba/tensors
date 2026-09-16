"""Loss functions, split into the phases a training step actually pays.

A public cross-entropy call does three separable things: it validates or
expands its targets, it evaluates the fused loss, and it reduces. Target
preparation is measured on its own because it is the part that depends on
how the caller supplies labels — class indices have to be expanded into a
dense distribution, and a dense distribution has to be validated.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import (
    execute_binary_cross_entropy,
    execute_binary_cross_entropy_gradient,
    execute_cross_entropy,
    execute_cross_entropy_gradient,
    execute_one_hot_targets,
    execute_validate_distributions,
)

from ..case import Case, Group, Unsupported
from ..workloads import ACCELERATED, FLOAT_DTYPES, dtype_of, tensor


def _dense_targets(
    batch: int, classes: int, dtype_name: str
) -> ts.Tensor:
    """Return a valid dense probability distribution per row."""
    row = [1.0 / classes] * classes
    return ts.Tensor(
        row * batch, dtype=dtype_of(dtype_name), shape=(batch, classes)
    )


def _index_targets(batch: int, classes: int) -> ts.Tensor:
    """Return class indices, as a caller supplying labels would."""
    return ts.Tensor(
        [float(index % classes) for index in range(batch)],
        dtype=ts.float64,
        shape=(batch,),
    )


def _cross_entropy_cases(
    backend: str,
    batch: int,
    classes: int,
    dtype_name: str,
    reduction: str,
) -> list[Case]:
    """Build target-preparation, kernel, public, and backward cases."""
    dtype = dtype_of(dtype_name)
    elements = batch * classes
    suffix = f"{dtype_name}/{batch}x{classes}/{reduction}"
    output_shape = (batch,) if reduction == "none" else (1,)
    common: dict[str, Any] = {
        "family": "loss/cross_entropy",
        "dtype": dtype_name,
        "shape": (batch, classes),
        "elements": elements,
        "work_items": elements,
        "tags": {
            "ladder": f"cross_entropy|{suffix}",
            "curve": f"cross_entropy|{dtype_name}|{reduction}",
            "dtype_pair": f"cross_entropy|{reduction}",
            "pair": f"cross_entropy|{suffix}",
            "phase": "forward",
        },
    }
    cases: list[Case] = []
    logits = tensor((batch, classes), dtype_name=dtype_name, kind="ramp")
    dense = _dense_targets(batch, classes, dtype_name)
    indices = _index_targets(batch, classes)
    # The kernels index ``logits.shape[:axis]`` directly, so they take
    # the resolved axis rather than the public ``-1`` spelling.
    axis = logits.ndim - 1

    if backend in ACCELERATED:
        preparation_common = dict(common)
        preparation_common["tags"] = dict(common["tags"])
        preparation_common["tags"].pop("ladder", None)
        preparation_common["tags"].pop("pair", None)

        def run_one_hot() -> Any:
            return execute_one_hot_targets(logits, indices, axis)

        def validate_one_hot() -> None:
            if run_one_hot() is None:
                raise Unsupported(
                    "dense target expansion declined this configuration"
                )

        cases.append(Case(
            name=f"prepare.one_hot/{suffix}",
            run=run_one_hot,
            layer="dispatch",
            validate=validate_one_hot,
            description=(
                "expand class indices into a dense distribution, including "
                "the host-visible validity check"
            ),
            backends=ACCELERATED,
            **preparation_common,
        ))
        cases.append(Case(
            name=f"prepare.validate_distributions/{suffix}",
            run=lambda: execute_validate_distributions(dense, axis),
            layer="dispatch",
            validate=lambda: execute_validate_distributions(dense, axis),
            description=(
                "validate already-dense target rows, which reads a "
                "provider result back to the host"
            ),
            backends=ACCELERATED,
            **preparation_common,
        ))

        def run_dispatch() -> Any:
            return execute_cross_entropy(
                logits, dense, axis,
                reduction=reduction, dtype=dtype,
                output_shape=output_shape,
            )

        def validate_dispatch() -> None:
            if run_dispatch() is None:
                raise Unsupported(
                    "the fused cross-entropy kernel declined this "
                    "configuration and deferred to the reference path"
                )

        cases.append(Case(
            name=f"dispatch.cross_entropy/{suffix}",
            run=run_dispatch,
            layer="dispatch",
            validate=validate_dispatch,
            description="the fused dense cross-entropy kernel",
            backends=ACCELERATED,
            **common,
        ))

        gradient = tensor(output_shape, dtype_name=dtype_name, kind="constant")
        backward_common = dict(common)
        backward_common["tags"] = dict(common["tags"])
        backward_common["tags"]["phase"] = "backward"
        backward_common["tags"]["ladder"] = f"cross_entropy-vjp|{suffix}"

        def run_gradient() -> Any:
            return execute_cross_entropy_gradient(
                gradient, logits, dense, axis, reduction=reduction
            )

        def validate_gradient() -> None:
            if run_gradient() is None:
                raise Unsupported(
                    "the cross-entropy VJP declined this configuration"
                )

        cases.append(Case(
            name=f"vjp.cross_entropy/{suffix}",
            run=run_gradient,
            layer="dispatch",
            validate=validate_gradient,
            description="the fused cross-entropy VJP",
            backends=ACCELERATED,
            **backward_common,
        ))

    cases.append(Case(
        name=f"public.cross_entropy_dense/{suffix}",
        run=lambda: ts.cross_entropy(logits, dense, reduction=reduction),
        layer="public",
        validate=lambda: ts.cross_entropy(
            logits, dense, reduction=reduction
        ),
        description="public cross-entropy over dense target rows",
        **common,
    ))
    index_common = dict(common)
    index_common["tags"] = dict(common["tags"])
    index_common["tags"].pop("ladder", None)
    cases.append(Case(
        name=f"public.cross_entropy_indices/{suffix}",
        run=lambda: ts.cross_entropy(logits, indices, reduction=reduction),
        layer="public",
        validate=lambda: ts.cross_entropy(
            logits, indices, reduction=reduction
        ),
        description=(
            "public cross-entropy over class indices, which includes dense "
            "target expansion"
        ),
        **index_common,
    ))

    # The differentiated public path, so backward can be compared against
    # the forward loss it differentiates.
    logits_variable = ts.Variable(logits)
    dense_variable = ts.Variable(dense, requires_grad=False)
    autograd_tags = dict(common["tags"])
    autograd_tags["ladder"] = f"cross_entropy-autograd|{suffix}"
    autograd_tags["phase"] = "backward"
    autograd_common = dict(common)
    autograd_common["tags"] = autograd_tags

    def run_backward() -> None:
        loss = ts.cross_entropy(
            logits_variable, dense_variable, reduction=reduction
        )
        ts.backward(loss)

    cases.append(Case(
        name=f"autograd.cross_entropy/{suffix}",
        run=run_backward,
        layer="autograd",
        validate=run_backward,
        description="trace, forward, and differentiate a public loss",
        gc_enabled=True,
        **autograd_common,
    ))
    return cases


def _binary_cases(
    backend: str,
    size: int,
    dtype_name: str,
    from_logits: bool,
) -> list[Case]:
    """Build binary cross-entropy cases in both input conventions."""
    dtype = dtype_of(dtype_name)
    shape = (size,)
    convention = "logits" if from_logits else "probabilities"
    suffix = f"{dtype_name}/{size}/{convention}"
    common: dict[str, Any] = {
        "family": "loss/binary_cross_entropy",
        "dtype": dtype_name,
        "shape": shape,
        "elements": size,
        "work_items": size,
        "tags": {
            "ladder": f"binary_cross_entropy|{suffix}",
            "curve": f"binary_cross_entropy|{dtype_name}|{convention}",
            "dtype_pair": f"binary_cross_entropy|{convention}",
            "pair": f"binary_cross_entropy|{suffix}",
            "phase": "forward",
        },
    }
    cases: list[Case] = []
    # Probabilities must stay inside the open unit interval; logits are free.
    prediction = (
        tensor(shape, dtype_name=dtype_name, kind="ramp")
        if from_logits
        else ts.Tensor(
            [0.25 + (index % 5) / 10.0 for index in range(size)],
            dtype=dtype, shape=shape,
        )
    )
    target = ts.Tensor(
        [float(index % 2) for index in range(size)], dtype=dtype, shape=shape
    )

    if backend in ACCELERATED:
        def run_dispatch() -> Any:
            return execute_binary_cross_entropy(
                prediction, target,
                from_logits=from_logits, reduction="mean",
                dtype=dtype, output_shape=(1,),
            )

        def validate_dispatch() -> None:
            if run_dispatch() is None:
                raise Unsupported(
                    "the fused binary cross-entropy kernel declined this "
                    "configuration"
                )

        cases.append(Case(
            name=f"dispatch.binary_cross_entropy/{suffix}",
            run=run_dispatch,
            layer="dispatch",
            validate=validate_dispatch,
            description="the fused binary cross-entropy kernel",
            backends=ACCELERATED,
            **common,
        ))

        gradient = tensor((1,), dtype_name=dtype_name, kind="constant")
        backward_common = dict(common)
        backward_common["tags"] = dict(common["tags"])
        backward_common["tags"]["phase"] = "backward"
        backward_common["tags"]["ladder"] = (
            f"binary_cross_entropy-vjp|{suffix}"
        )

        def run_gradient() -> Any:
            return execute_binary_cross_entropy_gradient(
                gradient, prediction, target,
                from_logits=from_logits, reduction="mean",
            )

        def validate_gradient() -> None:
            if run_gradient() is None:
                raise Unsupported(
                    "the binary cross-entropy VJP declined this "
                    "configuration"
                )

        cases.append(Case(
            name=f"vjp.binary_cross_entropy/{suffix}",
            run=run_gradient,
            layer="dispatch",
            validate=validate_gradient,
            description="the fused binary cross-entropy VJP",
            backends=ACCELERATED,
            **backward_common,
        ))

    cases.append(Case(
        name=f"public.binary_cross_entropy/{suffix}",
        run=lambda: ts.binary_cross_entropy(
            prediction, target, from_logits=from_logits
        ),
        layer="public",
        validate=lambda: ts.binary_cross_entropy(
            prediction, target, from_logits=from_logits
        ),
        description="public binary cross-entropy",
        **common,
    ))
    return cases


def groups() -> list[Group]:
    """Return loss groups over batch, class count, dtype, and reduction."""
    result: list[Group] = []

    for dtype_name in FLOAT_DTYPES:
        for batch, classes in (
            (1, 10), (32, 10), (128, 1_000), (512, 1_000), (64, 10_000),
        ):
            for reduction in ("mean", "none"):
                elements = batch * classes

                def factory(
                    backend: str,
                    batch: int = batch,
                    classes: int = classes,
                    dtype_name: str = dtype_name,
                    reduction: str = reduction,
                    elements: int = elements,
                ) -> Sequence[Case]:
                    if backend == "python" and elements > 20_000:
                        raise Unsupported(
                            "exceeds the Python backend loss ceiling"
                        )
                    return _cross_entropy_cases(
                        backend, batch, classes, dtype_name, reduction
                    )

                result.append(Group(
                    name=(
                        f"losses/cross_entropy/{dtype_name}/"
                        f"{batch}x{classes}/{reduction}"
                    ),
                    factory=factory,
                    suite="losses",
                ))

    for dtype_name in FLOAT_DTYPES:
        for size in (100, 10_000, 1_000_000):
            for from_logits in (False, True):
                def binary_factory(
                    backend: str,
                    size: int = size,
                    dtype_name: str = dtype_name,
                    from_logits: bool = from_logits,
                ) -> Sequence[Case]:
                    if backend == "python" and size > 20_000:
                        raise Unsupported(
                            "exceeds the Python backend loss ceiling"
                        )
                    return _binary_cases(
                        backend, size, dtype_name, from_logits
                    )

                convention = "logits" if from_logits else "probabilities"
                result.append(Group(
                    name=(
                        f"losses/binary/{dtype_name}/{size}/{convention}"
                    ),
                    factory=binary_factory,
                    suite="losses",
                ))
    return result


__all__ = ["groups"]
