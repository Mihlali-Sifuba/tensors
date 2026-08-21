"""Standard-deviation public API."""

import math as _math
from typing import Any, List

from ..dtype import float64
from ..tensor import Tensor
from ._reduction import Axis, immutable_axis, normalize_axes, reduction_groups


def _scaled_deviations(
    value: Tensor,
    group: list[int],
) -> tuple[float, list[float], float]:
    """Return a safe scale, centered scaled values, and their deviation."""
    values = [float(value._data[index]) for index in group]
    if any(not _math.isfinite(item) for item in values):
        return _math.nan, [_math.nan] * len(values), _math.nan

    count = len(values)
    average = _math.fsum(item / count for item in values)
    centered = [item - average for item in values]
    if all(_math.isfinite(item) for item in centered):
        scale = max((abs(item) for item in centered), default=0.0)
        if scale == 0.0:
            return 0.0, [0.0] * count, 0.0
        normalized_centered = [item / scale for item in centered]
    else:
        scale = max((abs(item) for item in values), default=0.0)
        normalized = [item / scale for item in values]
        normalized_average = _math.fsum(item / count for item in normalized)
        normalized_centered = [item - normalized_average for item in normalized]

    variance = _math.fsum(
        item * item / count for item in normalized_centered
    )
    return scale, normalized_centered, _math.sqrt(variance)


class Std:
    """Population standard-deviation operation."""

    @staticmethod
    def forward(
        value: Tensor,
        axis: Axis = None,
        keepdims: bool = False,
    ) -> Tensor:
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        values = []
        for group in groups:
            if not group:
                values.append(0.0)
                continue
            scale, _, normalized_deviation = _scaled_deviations(value, group)
            values.append(scale * normalized_deviation)
        dtype = value.dtype if value.dtype.typecode in {"f", "d"} else float64
        return Tensor(values, dtype=dtype, shape=output_shape)

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        """Differentiate the population standard deviation by reduction group."""
        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        _, output_shape, groups = reduction_groups(
            value, axis, keepdims, scalar_as_vector=True
        )
        if grad.shape != output_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {output_shape}"
            )
        result = [0.0] * value.size
        for output_index, group in enumerate(groups):
            if not group:
                continue
            _, centered, normalized_deviation = _scaled_deviations(value, group)
            if normalized_deviation == 0:
                continue
            normalizer = len(group) * normalized_deviation
            upstream = grad._data[output_index]
            for input_index, centered_value in zip(group, centered):
                result[input_index] = upstream * (centered_value / normalizer)
        return [Tensor(result, dtype=grad.dtype, shape=value.shape)]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable population-standard-deviation VJP."""
        from .mean import mean
        from .reshape import reshape
        from ..variable import Variable

        value = inputs[0]
        axis = kwargs.get("axis")
        keepdims = bool(kwargs.get("keepdims", False))
        _, scale_shape, groups = reduction_groups(value.data, axis, True)
        statistics = [_scaled_deviations(value.data, group) for group in groups]
        if any(
            group and normalized_deviation == 0
            for group, (_, _, normalized_deviation) in zip(groups, statistics)
        ):
            raise ValueError(
                "Higher-order derivatives of std are undefined at zero deviation"
            )
        count = len(groups[0]) if groups else 0
        if count == 0:
            raise NotImplementedError(
                "Higher-order derivatives for empty standard deviations are not implemented"
            )
        scales = Variable(
            Tensor(
                [
                    scale if _math.isfinite(scale) and scale > 0.0 else 1.0
                    for scale, _, _ in statistics
                ],
                dtype=value.dtype,
                shape=scale_shape,
            ),
            requires_grad=False,
        )
        normalized = value / scales
        center = mean(normalized, axis=axis, keepdims=True)
        deviation = std(normalized, axis=axis, keepdims=True)
        expanded = grad if keepdims else reshape(
            grad,
            tuple(1 if index in normalize_axes(value.ndim, axis) else size
                  for index, size in enumerate(value.shape)),
        )
        return [expanded * (normalized - center) / (count * deviation)]


def std(value: Any, axis: Axis = None, keepdims: bool = False) -> Any:
    """Compute population standard deviation over selected axes."""
    from ..variable import Variable

    axis = immutable_axis(axis)

    if isinstance(value, Variable):
        return Variable._from_operation(
            Std.forward(value.data, axis=axis, keepdims=keepdims),
            "std",
            Std,
            [value],
            axis=axis,
            keepdims=keepdims,
        )
    if not isinstance(value, Tensor):
        value = Tensor(value)
    return Std.forward(value, axis=axis, keepdims=keepdims)


__all__ = ["Std", "std"]
