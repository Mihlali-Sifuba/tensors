"""Numerically stable multiclass cross-entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Literal, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape, sum_to_shape_graph
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_shape, broadcast_tensors
from ..utils.shape import (
    coordinates_to_index,
    index_to_coordinates,
    shape_size,
)
from ._reduction import keepdims_shape, reduction_groups
from .log_softmax import LogSoftmax, log_softmax
from .mean import _stable_float_mean
from ._normalization import shifted_normalization
from .softmax import Softmax, _normalize_axis
from .sum import _stable_float_sum

if TYPE_CHECKING:
    from ..variable import Variable


Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> None:
    if not isinstance(reduction, str):
        raise TypeError("reduction must be a string")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _tensor(value: Any) -> Tensor:
    from ..variable import Variable

    if isinstance(value, Variable):
        return value.data
    return value if isinstance(value, Tensor) else Tensor(value)


def _one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Tensor:
    """Expand one class index per sample into dense target distributions."""
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
    sample_count = shape_size(sample_shape)
    scalar_target = sample_shape == () and targets.size == 1
    if targets.shape != sample_shape and not scalar_target:
        raise ValueError(
            f"Class-index target shape {targets.shape} does not match "
            f"logits sample shape {sample_shape}"
        )

    values = [0.0] * logits.size
    class_count = logits.shape[axis]
    for sample_index in range(sample_count):
        target = float(targets._data[sample_index])
        if not math.isfinite(target) or not target.is_integer():
            raise ValueError("Class-index targets must contain integers")
        class_index = int(target)
        if not 0 <= class_index < class_count:
            raise ValueError(
                f"Class index {class_index} is outside [0, {class_count})"
            )
        sample_coordinates = index_to_coordinates(sample_index, sample_shape)
        coordinates = (
            sample_coordinates[:axis]
            + (class_index,)
            + sample_coordinates[axis:]
        )
        values[coordinates_to_index(coordinates, logits.shape)] = 1.0
    return Tensor(values, dtype=logits.dtype, shape=logits.shape)


def _targets_for_logits(logits: Any, targets: Any, axis: int) -> Any:
    """Return dense targets, retaining Variables when targets are trainable."""
    from ..variable import Variable

    logits_tensor = _tensor(logits)
    target_tensor = _tensor(targets)
    sample_shape = logits_tensor.shape[:axis] + logits_tensor.shape[axis + 1:]
    scalar_target = sample_shape == () and target_tensor.size == 1

    if target_tensor.shape == logits_tensor.shape:
        prepared = targets
    elif target_tensor.shape == sample_shape or scalar_target:
        dense_target = False
        if target_tensor.dtype.kind == "floating":
            try:
                target_shape = broadcast_shape(
                    target_tensor.shape,
                    logits_tensor.shape,
                )
            except ValueError:
                target_shape = None

            if target_shape == logits_tensor.shape:
                values = [float(value) for value in target_tensor._data]
                dense_target = any(
                    not math.isfinite(value) or not value.is_integer()
                    for value in values
                )
                if not dense_target:
                    _, expanded_targets = broadcast_tensors(
                        logits_tensor,
                        target_tensor,
                    )
                    try:
                        _validate_distributions(expanded_targets, axis)
                    except ValueError:
                        pass
                    else:
                        dense_target = True

        if dense_target:
            prepared = targets
        else:
            if isinstance(targets, Variable):
                raise TypeError(
                    "Class-index targets cannot be differentiable Variables"
                )
            prepared = _one_hot_targets(logits_tensor, target_tensor, axis)
    else:
        try:
            target_shape = broadcast_shape(
                target_tensor.shape,
                logits_tensor.shape,
            )
        except ValueError as exc:
            raise ValueError(
                f"Target shape {target_tensor.shape} is neither class-index shaped "
                f"nor broadcastable to logits shape {logits_tensor.shape}"
            ) from exc
        if target_shape != logits_tensor.shape:
            raise ValueError(
                f"Target shape {target_tensor.shape} cannot broadcast to "
                f"logits shape {logits_tensor.shape}"
            )
        prepared = targets

    if isinstance(logits, Variable) and not isinstance(prepared, Variable):
        return Variable(_tensor(prepared), requires_grad=False)
    if isinstance(prepared, Variable):
        return prepared
    return _tensor(prepared)


def _validate_distributions(targets: Tensor, axis: int) -> None:
    """Require finite probability distributions along the class axis."""
    _, _, groups = reduction_groups(targets, axis, keepdims=False)
    for group in groups:
        values = [float(targets._data[index]) for index in group]
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
            raise ValueError(
                "Dense cross-entropy targets must contain probabilities "
                "between 0 and 1"
            )
        if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError(
                "Dense cross-entropy targets must sum to 1 along the class axis"
            )


class CrossEntropy:
    """Stable cross-entropy between logits and dense target distributions."""

    @staticmethod
    def forward(
        logits: Tensor,
        targets: Tensor,
        *,
        axis: int = -1,
        reduction: Reduction = "mean",
    ) -> Tensor:
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)
        logits, targets = broadcast_tensors(logits, targets)
        _validate_distributions(targets, axis)
        log_probabilities = LogSoftmax.forward(logits, axis=axis)
        _, output_shape, groups = reduction_groups(
            logits,
            axis,
            keepdims=False,
        )

        # Zero-probability targets make no contribution. Skipping those terms
        # avoids the otherwise indeterminate floating-point product 0 * -inf.
        losses = []
        for group in groups:
            contributions = [
                -float(targets._data[index]) * log_probabilities._data[index]
                for index in group
                if targets._data[index] != 0
            ]
            losses.append(_stable_float_sum(contributions))

        dtype = result_dtype(logits.dtype, targets, division=True)
        if reduction == "none":
            return Tensor(losses, dtype=dtype, shape=output_shape)
        if reduction == "mean":
            total = _stable_float_mean(losses)
        else:
            total = _stable_float_sum(losses)
        return Tensor([total], dtype=dtype, shape=(1,))

    @staticmethod
    def backward(grad: Tensor, *inputs: Tensor, **kwargs: object) -> List[Tensor]:
        logits, targets = inputs
        axis = kwargs.get("axis", -1)
        reduction = kwargs.get("reduction", "mean")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)

        expanded_logits, expanded_targets = broadcast_tensors(logits, targets)
        _validate_distributions(expanded_targets, axis)
        probabilities = Softmax.forward(expanded_logits, axis=axis)
        log_probabilities = LogSoftmax.forward(expanded_logits, axis=axis)
        _, output_shape, groups = reduction_groups(
            expanded_logits,
            axis,
            keepdims=False,
        )
        expected_shape = output_shape if reduction == "none" else (1,)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{expected_shape}"
            )

        logits_gradient = [0.0] * expanded_logits.size
        targets_gradient = [0.0] * expanded_targets.size
        for output_index, group in enumerate(groups):
            if reduction == "none":
                upstream = grad._data[output_index]
            else:
                upstream = grad._data[0]
                if reduction == "mean" and groups:
                    upstream /= len(groups)
            if upstream == 0:
                continue
            target_mass = math.fsum(
                float(expanded_targets._data[index]) for index in group
            )
            group_values = [
                float(expanded_logits._data[index]) for index in group
            ]
            if group_values and all(math.isfinite(value) for value in group_values):
                _, _, _, complements = shifted_normalization(group_values)
            else:
                complements = [
                    1.0 - float(probabilities._data[index]) for index in group
                ]
            for index, complement in zip(group, complements):
                probability = float(probabilities._data[index])
                target_value = float(expanded_targets._data[index])
                if probability > 0.5:
                    derivative = _stable_float_sum([
                        target_mass - target_value,
                        -target_mass * complement,
                    ])
                else:
                    derivative = target_mass * probability - target_value
                logits_gradient[index] = upstream * derivative
                targets_gradient[index] = (
                    -upstream * log_probabilities._data[index]
                )

        expanded_shape = expanded_logits.shape
        return [
            sum_to_shape(
                Tensor(logits_gradient, dtype=grad.dtype, shape=expanded_shape),
                logits.shape,
            ),
            sum_to_shape(
                Tensor(targets_gradient, dtype=grad.dtype, shape=expanded_shape),
                targets.shape,
            ),
        ]

    @staticmethod
    def backward_graph(grad, *inputs, **kwargs: object):
        """Build a differentiable cross-entropy VJP."""
        from .log_softmax import _log_softmax_vjp
        from .reshape import reshape

        logits, targets = inputs
        axis = kwargs.get("axis", -1)
        reduction = kwargs.get("reduction", "mean")
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits.data, axis)

        # Multiplying by finite ones expands targets without evaluating the
        # indeterminate expression ``infinite_logits * 0``. The logits VJP is
        # the log-softmax VJP seeded by ``-targets``; its dedicated operation
        # retains tiny dominant-class derivatives that ordinary subtraction
        # would round away.
        ones = Tensor(
            [1.0] * logits.size,
            dtype=targets.dtype,
            shape=logits.shape,
        )
        expanded_targets = targets * ones
        logits_derivative = _log_softmax_vjp(
            -expanded_targets,
            logits,
            axis,
        )
        targets_derivative = -log_softmax(logits, axis=axis) + expanded_targets * 0.0

        if reduction == "none":
            upstream = reshape(grad, keepdims_shape(logits.shape, axis))
        else:
            upstream = grad
            if reduction == "mean":
                sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
                sample_count = shape_size(sample_shape)
                if sample_count:
                    upstream = upstream / sample_count

        return [
            sum_to_shape_graph(upstream * logits_derivative, logits.shape),
            sum_to_shape_graph(upstream * targets_derivative, targets.shape),
        ]


@overload
def cross_entropy(
    logits: Variable,
    targets: TensorLike,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def cross_entropy(
    logits: TensorLike,
    targets: Variable,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> Variable: ...


@overload
def cross_entropy(
    logits: TensorData,
    targets: TensorData,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> Tensor: ...


def cross_entropy(
    logits: TensorLike,
    targets: TensorLike,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> TensorResult:
    """Return stable multiclass cross-entropy from unnormalized ``logits``.

    ``targets`` may contain class indices with the class axis removed, or
    dense probability distributions broadcastable to the logits shape.
    In an otherwise ambiguous shape, an integer dtype selects class indices
    while a floating probability distribution selects dense targets.
    """
    _validate_reduction(reduction)
    from ..variable import Variable

    logits_tensor = _tensor(logits)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("cross_entropy axis must be an integer")
    axis = _normalize_axis(logits_tensor, axis)
    prepared_targets = _targets_for_logits(logits, targets, axis)

    logits_is_variable = isinstance(logits, Variable)
    targets_are_variable = isinstance(prepared_targets, Variable)
    if logits_is_variable or targets_are_variable:
        logits_variable = logits if logits_is_variable else Variable(
            logits_tensor,
            requires_grad=False,
        )
        targets_variable = (
            prepared_targets
            if targets_are_variable
            else Variable(_tensor(prepared_targets), requires_grad=False)
        )
        return Variable._from_operation(
            CrossEntropy.forward(
                logits_variable.data,
                targets_variable.data,
                axis=axis,
                reduction=reduction,
            ),
            "cross_entropy",
            CrossEntropy,
            [logits_variable, targets_variable],
            axis=axis,
            reduction=reduction,
        )

    return CrossEntropy.forward(
        logits_tensor,
        _tensor(prepared_targets),
        axis=axis,
        reduction=reduction,
    )


__all__ = ["CrossEntropy", "Reduction", "cross_entropy"]
