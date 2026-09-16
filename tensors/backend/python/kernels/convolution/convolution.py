"""Reference grouped cross-correlation for the Python backend."""

from __future__ import annotations
from typing import Any
from tensors.backend.python.storage import PythonStorage
from tensors.math.sum import _stable_product_sum, _stable_float_sum


def convolution(
    inputs, kernel, bias, *, dtype, output_shape, stride, padding, dilation, groups
):
    """Accumulate each receptive field with reference-exact arithmetic."""
    from tensors.math.convolution import _geometry, _contributions

    geometry = _geometry(
        len(stride), inputs, kernel, bias, stride, padding, dilation, groups
    )
    input_data = inputs._data
    kernel_data = kernel._data
    bias_data = bias._data if bias is not None else None
    exact = dtype.kind == "integer"
    values: list[Any] = []
    for _, out_channel, pairs in _contributions(geometry, inputs, kernel):
        total: Any
        if exact:
            total = sum(
                (
                    int(input_data[source]) * int(kernel_data[weight])
                    for source, weight in pairs
                )
            )
            if bias_data is not None:
                total += int(bias_data[out_channel])
        else:
            total = _stable_product_sum(
                [
                    (float(input_data[source]), float(kernel_data[weight]))
                    for source, weight in pairs
                ]
            )
            if bias_data is not None:
                total = _stable_float_sum([total, float(bias_data[out_channel])])
        values.append(total)
    return PythonStorage.from_values(values, dtype)
