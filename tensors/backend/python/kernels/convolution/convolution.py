"""Reference grouped cross-correlation for the Python backend."""

from __future__ import annotations
from typing import Any
from tensors.backend.python.storage import PythonStorage
from tensors.utils.convolution import (
    contributions as _contributions,
    resolve_geometry,
)
from tensors.utils.summation import stable_float_sum, stable_product_sum


def convolution(
    input_values,
    kernel_values,
    bias_values,
    input_shape,
    kernel_shape,
    bias_shape,
    *,
    dtype,
    output_shape,
    stride,
    padding,
    dilation,
    groups,
):
    """Accumulate each receptive field with reference-exact arithmetic."""

    geometry = resolve_geometry(
        len(stride),
        input_shape,
        kernel_shape,
        bias_shape,
        stride,
        padding,
        dilation,
        groups,
    )
    exact = dtype.kind == "integer"
    values: list[Any] = []
    for _, out_channel, pairs in _contributions(geometry, kernel_shape):
        total: Any
        if exact:
            total = sum(
                (
                    int(input_values[source]) * int(kernel_values[weight])
                    for source, weight in pairs
                )
            )
            if bias_values is not None:
                total += int(bias_values[out_channel])
        else:
            total = stable_product_sum(
                [
                    (float(input_values[source]), float(kernel_values[weight]))
                    for source, weight in pairs
                ]
            )
            if bias_values is not None:
                total = stable_float_sum([total, float(bias_values[out_channel])])
        values.append(total)
    return PythonStorage.from_values(values, dtype)
