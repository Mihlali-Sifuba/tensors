"""Reference grouped cross-correlation VJPs for the Python backend."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tensors.backend.python.storage import PythonStorage
from tensors.utils.convolution import (
    contributions as _contributions,
    resolve_geometry,
)
from tensors.backend.storage import Storage
from tensors.utils.summation import stable_float_sum, stable_product_sum

if TYPE_CHECKING:
    from tensors.tensor import Tensor


def convolution_gradient(
    grad: Tensor,
    inputs: Tensor,
    kernel: Tensor,
    *,
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int,
    include_bias: bool,
    needs_input_grad: tuple[bool, ...],
) -> list[Storage | None]:
    """Collect the requested VJP terms before summing them stably.

    Every contribution to one output element is gathered before it is summed,
    so the result does not depend on the order the receptive fields are
    traversed in.
    """

    geometry = resolve_geometry(
        len(stride),
        inputs.shape,
        kernel.shape,
        None,
        stride,
        padding,
        dilation,
        groups,
    )
    need_values, need_kernel = (needs_input_grad[0], needs_input_grad[1])
    need_bias = include_bias and needs_input_grad[2]
    if not (need_values or need_kernel or need_bias):
        return [None] * len(needs_input_grad)

    grad_data = grad._data
    input_data = inputs._data
    kernel_data = kernel._data
    input_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(inputs.size if need_values else 0)
    ]
    kernel_terms: list[list[tuple[float, float]]] = [
        [] for _ in range(kernel.size if need_kernel else 0)
    ]
    bias_terms: list[list[float]] = [
        [] for _ in range(geometry.out_channels if need_bias else 0)
    ]
    for index, out_channel, pairs in _contributions(geometry, kernel.shape):
        upstream = float(grad_data[index])
        if need_values or need_kernel:
            for source, weight in pairs:
                if need_values:
                    input_terms[source].append((upstream, float(kernel_data[weight])))
                if need_kernel:
                    kernel_terms[weight].append((upstream, float(input_data[source])))
        if need_bias:
            bias_terms[out_channel].append(upstream)

    results: list[Storage | None] = [
        (
            PythonStorage.from_values(
                [stable_product_sum(terms) for terms in input_terms],
                grad.dtype,
            )
            if need_values
            else None
        ),
        (
            PythonStorage.from_values(
                [stable_product_sum(terms) for terms in kernel_terms],
                grad.dtype,
            )
            if need_kernel
            else None
        ),
    ]
    if include_bias:
        results.append(
            PythonStorage.from_values(
                [stable_float_sum(terms) for terms in bias_terms],
                grad.dtype,
            )
            if need_bias
            else None
        )
    return results
