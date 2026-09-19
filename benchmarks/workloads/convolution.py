"""Convolution across dimensionality, shape, and every geometry parameter.

Convolution has more independent knobs than any other operation here, and
they do not affect cost the same way: batch and channels scale the matrix
product, spatial extent and kernel size scale both the product and the
im2col expansion the kernel builds, and stride and padding change the output
extent without changing the input. Each knob is therefore varied on its own
against a fixed reference configuration, so a curve attributes cost to that
knob rather than to a shape change in general.

Forward and backward are separate cases over the same configuration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import tensors as ts
from tensors.backend import execute_convolution, execute_convolution_gradient

from ..case import Case, Group, Unsupported
from ..inputs import ACCELERATED, FLOAT_DTYPES, dtype_of, tensor

#: The public entry point for each rank.
_PUBLIC = {1: ts.conv1d, 2: ts.conv2d, 3: ts.conv3d}


def _output_extent(
    extent: int, kernel: int, stride: int, padding: int, dilation: int
) -> int:
    """Return one spatial dimension of a convolution's output."""
    effective = dilation * (kernel - 1) + 1
    return (extent + 2 * padding - effective) // stride + 1


class Configuration:
    """One convolution geometry, with the shapes it implies."""

    def __init__(
        self,
        *,
        rank: int,
        batch: int,
        in_channels: int,
        out_channels: int,
        extent: int,
        kernel: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        self.rank = rank
        self.batch = batch
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.extent = extent
        self.kernel = kernel
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias

    @property
    def input_shape(self) -> tuple[int, ...]:
        return (self.batch, self.in_channels) + (self.extent,) * self.rank

    @property
    def kernel_shape(self) -> tuple[int, ...]:
        return (
            self.out_channels,
            self.in_channels // self.groups,
        ) + (self.kernel,) * self.rank

    @property
    def output_extent(self) -> int:
        return _output_extent(
            self.extent,
            self.kernel,
            self.stride,
            self.padding,
            self.dilation,
        )

    @property
    def output_shape(self) -> tuple[int, ...]:
        return (
            self.batch,
            self.out_channels,
        ) + (self.output_extent,) * self.rank

    @property
    def elements(self) -> int:
        total = 1
        for dimension in self.output_shape:
            total *= dimension
        return total

    @property
    def multiply_accumulates(self) -> int:
        """Return the arithmetic the geometry implies, for throughput."""
        patch = (self.in_channels // self.groups) * self.kernel**self.rank
        return self.elements * patch

    @property
    def label(self) -> str:
        return (
            f"r{self.rank}/b{self.batch}/c{self.in_channels}-"
            f"{self.out_channels}/e{self.extent}/k{self.kernel}"
            f"/s{self.stride}/p{self.padding}/d{self.dilation}"
            f"/g{self.groups}"
        )

    def valid(self) -> bool:
        """Return whether this geometry produces a non-empty output."""
        return (
            self.output_extent > 0
            and self.in_channels % self.groups == 0
            and self.out_channels % self.groups == 0
        )


def _convolution_cases(
    backend: str,
    configuration: Configuration,
    dtype_name: str,
) -> list[Case]:
    """Build dispatch and public forward and backward cases."""
    if not configuration.valid():
        raise Unsupported(
            "this geometry produces an empty output or an invalid group "
            "division, so it is not a convolution the package accepts"
        )
    dtype = dtype_of(dtype_name)
    spatial = (configuration.stride,) * configuration.rank
    padding = (configuration.padding,) * configuration.rank
    dilation = (configuration.dilation,) * configuration.rank

    inputs = tensor(configuration.input_shape, dtype_name=dtype_name, kind="ramp")
    kernel = tensor(
        configuration.kernel_shape,
        dtype_name=dtype_name,
        kind="constant",
        value=0.1,
    )
    bias = (
        tensor(
            (configuration.out_channels,),
            dtype_name=dtype_name,
            kind="constant",
            value=0.01,
        )
        if configuration.bias
        else None
    )

    common: dict[str, Any] = {
        "family": f"conv{configuration.rank}d",
        "dtype": dtype_name,
        "shape": configuration.output_shape,
        "elements": configuration.elements,
        "work_items": configuration.multiply_accumulates,
        "tags": {
            "ladder": f"conv|{dtype_name}|{configuration.label}",
            "curve": f"conv{configuration.rank}d|{dtype_name}",
            "dtype_pair": f"conv{configuration.rank}d",
            "pair": f"conv|{dtype_name}|{configuration.label}",
            "rank": str(configuration.rank),
        },
    }
    forward_tags = dict(common["tags"])
    forward_tags["phase"] = "forward"
    backward_tags = dict(common["tags"])
    backward_tags["phase"] = "backward"
    backward_tags["ladder"] = f"conv-vjp|{dtype_name}|{configuration.label}"

    cases: list[Case] = []
    public = _PUBLIC[configuration.rank]

    def run_public() -> Any:
        return public(
            inputs,
            kernel,
            bias,
            stride=configuration.stride,
            padding=configuration.padding,
            dilation=configuration.dilation,
            groups=configuration.groups,
        )

    def validate_public() -> None:
        result = run_public()
        assert tuple(result.shape) == configuration.output_shape

    forward_common = dict(common)
    forward_common["tags"] = forward_tags
    cases.append(
        Case(
            name=f"public.conv{configuration.rank}d/{dtype_name}"
            f"/{configuration.label}",
            run=run_public,
            layer="public",
            validate=validate_public,
            description="public convolution forward",
            memory=True,
            **forward_common,
        )
    )

    if backend in ACCELERATED:

        def run_dispatch() -> Any:
            return execute_convolution(
                inputs,
                kernel,
                bias,
                dtype=dtype,
                output_shape=configuration.output_shape,
                stride=spatial,
                padding=padding,
                dilation=dilation,
                groups=configuration.groups,
            )

        def validate_dispatch() -> None:
            if run_dispatch() is None:
                raise Unsupported(
                    "the convolution dispatcher declined this configuration "
                    "and deferred to the Python reference implementation"
                )

        dispatch_common = dict(common)
        dispatch_common["tags"] = forward_tags
        cases.append(
            Case(
                name=f"dispatch.conv{configuration.rank}d/{dtype_name}"
                f"/{configuration.label}",
                run=run_dispatch,
                layer="dispatch",
                validate=validate_dispatch,
                description="execute_convolution: policy, im2col, and tiling",
                backends=ACCELERATED,
                **dispatch_common,
            )
        )

        gradient = tensor(
            configuration.output_shape,
            dtype_name=dtype_name,
            kind="constant",
            value=1.0,
        )

        def run_gradient() -> Any:
            return execute_convolution_gradient(
                gradient,
                inputs,
                kernel,
                stride=spatial,
                padding=padding,
                dilation=dilation,
                groups=configuration.groups,
                include_bias=configuration.bias,
            )

        def validate_gradient() -> None:
            if run_gradient() is None:
                raise Unsupported(
                    "the convolution VJP declined this configuration and "
                    "deferred to the Python reference implementation"
                )

        backward_common = dict(common)
        backward_common["tags"] = backward_tags
        cases.append(
            Case(
                name=f"vjp.conv{configuration.rank}d/{dtype_name}"
                f"/{configuration.label}",
                run=run_gradient,
                layer="dispatch",
                validate=validate_gradient,
                description="all three convolution VJPs through dispatch",
                backends=ACCELERATED,
                memory=True,
                **backward_common,
            )
        )

    return cases


#: The configuration every single-knob curve varies from.
def _reference(rank: int) -> dict[str, Any]:
    """Return the reference geometry for one rank."""
    extent = {1: 1_024, 2: 32, 3: 12}[rank]
    return {
        "rank": rank,
        "batch": 8,
        "in_channels": 16,
        "out_channels": 16,
        "extent": extent,
        "kernel": 3,
    }


#: Each knob and the values its curve sweeps.
CURVES: dict[str, tuple[str, tuple[Any, ...]]] = {
    "batch": ("batch", (1, 2, 8, 32, 128)),
    "in-channels": ("in_channels", (1, 4, 16, 64, 256)),
    "out-channels": ("out_channels", (1, 4, 16, 64, 256)),
    "kernel": ("kernel", (1, 3, 5, 7, 11)),
    "stride": ("stride", (1, 2, 4)),
    "padding": ("padding", (0, 1, 3)),
    "dilation": ("dilation", (1, 2, 4)),
    "groups": ("groups", (1, 2, 4, 16)),
}

#: Spatial extents per rank, chosen so the element counts stay comparable.
EXTENTS: dict[int, tuple[int, ...]] = {
    1: (16, 128, 1_024, 8_192),
    2: (8, 16, 32, 64, 128),
    3: (4, 8, 12, 20),
}


#: The largest im2col expansion each backend may build. This machine has
#: about 3.4 GB free on the device and about 6 GB free on the host, and a
#: measurement that pushes either into paging or into a fallback allocator
#: reports the allocator rather than the kernel.
EXPANSION_CEILING: dict[str, int] = {
    "python": 8_000_000,
    "numpy": 1_200_000_000,
    "cuda": 700_000_000,
}


def _ceiling(backend: str, configuration: Configuration) -> None:
    """Reject a configuration too large for a backend, with a reason."""
    work = configuration.multiply_accumulates
    if backend == "python" and work > 400_000:
        raise Unsupported(
            f"{work} multiply-accumulates exceeds the Python backend "
            "convolution ceiling; the reference implementation accumulates "
            "element by element"
        )
    # The kernel materializes an im2col expansion whose width is the patch
    # size, so memory is the binding constraint rather than the output size.
    patch = (
        configuration.in_channels // configuration.groups
    ) * configuration.kernel**configuration.rank
    expansion_bytes = configuration.elements * patch * 8
    limit = EXPANSION_CEILING[backend]
    if expansion_bytes > limit:
        raise Unsupported(
            f"the im2col expansion would need about "
            f"{expansion_bytes / 1e9:.2f} GB, above the {limit / 1e9:.2f} GB "
            f"{backend} ceiling for this run; the kernel materializes an "
            "expansion of patch-size width, so memory bounds the geometry "
            "rather than the output"
        )


def groups() -> list[Group]:
    """Return convolution groups: rank curves, knob curves, and dtypes."""
    result: list[Group] = []

    # -- spatial extent per rank ----------------------------------------
    for rank, extents in EXTENTS.items():
        for extent in extents:
            settings = _reference(rank)
            settings["extent"] = extent

            def factory(
                backend: str,
                settings: dict[str, Any] = settings,
            ) -> Sequence[Case]:
                configuration = Configuration(**settings)
                _ceiling(backend, configuration)
                return _convolution_cases(backend, configuration, "float64")

            result.append(
                Group(
                    name=f"convolution/extent/conv{rank}d/{extent}",
                    factory=factory,
                    suite="convolution",
                )
            )

    # -- one curve per geometry knob, on 2-D --------------------------
    for curve_name, (attribute, values) in CURVES.items():
        for value in values:
            settings = _reference(2)
            settings[attribute] = value
            if attribute == "groups":
                # Grouping must divide both channel counts. Thirty-two is
                # the widest that keeps the ungrouped baseline inside the
                # expansion ceiling, so the curve has something to compare
                # against.
                settings["in_channels"] = 32
                settings["out_channels"] = 32

            def factory(
                backend: str,
                settings: dict[str, Any] = settings,
            ) -> Sequence[Case]:
                configuration = Configuration(**settings)
                _ceiling(backend, configuration)
                return _convolution_cases(backend, configuration, "float64")

            result.append(
                Group(
                    name=f"convolution/{curve_name}/conv2d/{value}",
                    factory=factory,
                    suite="convolution",
                )
            )

    # -- float32 against float64 on the reference geometry -------------
    for rank in (1, 2, 3):
        for dtype_name in FLOAT_DTYPES:
            settings = _reference(rank)

            def factory(
                backend: str,
                settings: dict[str, Any] = settings,
                dtype_name: str = dtype_name,
            ) -> Sequence[Case]:
                configuration = Configuration(**settings)
                _ceiling(backend, configuration)
                return _convolution_cases(backend, configuration, dtype_name)

            result.append(
                Group(
                    name=f"convolution/dtype/conv{rank}d/{dtype_name}",
                    factory=factory,
                    suite="convolution",
                )
            )

    # -- a realistic image-model layer ----------------------------------
    # Batch sizes here are chosen so the im2col expansion fits the device
    # ceiling above; the layer geometry is what these cases are for.
    for name, settings in {
        "resnet-stem": dict(
            rank=2,
            batch=2,
            in_channels=3,
            out_channels=64,
            extent=112,
            kernel=7,
            stride=2,
            padding=3,
        ),
        "resnet-block": dict(
            rank=2,
            batch=2,
            in_channels=64,
            out_channels=64,
            extent=28,
            kernel=3,
            stride=1,
            padding=1,
        ),
        "pointwise": dict(
            rank=2,
            batch=4,
            in_channels=128,
            out_channels=128,
            extent=28,
            kernel=1,
        ),
        "depthwise": dict(
            rank=2,
            batch=8,
            in_channels=64,
            out_channels=64,
            extent=56,
            kernel=3,
            padding=1,
            groups=64,
        ),
    }.items():

        def factory(
            backend: str,
            settings: dict[str, Any] = settings,
        ) -> Sequence[Case]:
            configuration = Configuration(**settings)
            _ceiling(backend, configuration)
            return _convolution_cases(backend, configuration, "float32")

        result.append(
            Group(
                name=f"convolution/realistic/{name}",
                factory=factory,
                suite="convolution",
            )
        )

    return result


__all__ = ["groups"]
