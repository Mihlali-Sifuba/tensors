"""Numerically stable multiclass cross-entropy."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, List, Literal, Optional, overload

from .._typing import TensorData, TensorLike, TensorResult
from ..backend import (
    execute_cross_entropy,
    execute_cross_entropy_gradient,
    execute_one_hot_targets,
    execute_validate_distributions,
)
from ..dtype import result_dtype
from ..ops._utils import sum_to_shape, sum_to_shape_graph
from ..shape import Shape
from ..ops.operation import Operation
from ..tensor import Tensor
from ..utils.broadcasting import broadcast_tensors
from ..utils.coordinates import (
    coordinates_to_linear_index,
    linear_index_to_coordinates,
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
    sample_count = Shape.from_iterable(sample_shape).size
    scalar_target = sample_shape == () and targets.size == 1
    if targets.shape != sample_shape and not scalar_target:
        raise ValueError(
            f"Class-index target shape {targets.shape} does not match "
            f"logits sample shape {sample_shape}"
        )

    accelerated = execute_one_hot_targets(logits, targets, axis)
    if accelerated is not None:
        return Tensor._from_owned_storage(
            accelerated,
            dtype=logits.dtype,
            shape=logits.shape,
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
        sample_coordinates = linear_index_to_coordinates(
            sample_index,
            sample_shape,
        )
        coordinates = (
            sample_coordinates[:axis]
            + (class_index,)
            + sample_coordinates[axis:]
        )
        values[
            coordinates_to_linear_index(coordinates, logits.shape)
        ] = 1.0
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
                target_shape = target_tensor.shape.broadcast_with(
                    logits_tensor.shape
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
            target_shape = target_tensor.shape.broadcast_with(
                logits_tensor.shape
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
    if execute_validate_distributions(targets, axis) is True:
        return
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


class CrossEntropy(Operation):
    """Stable cross-entropy between logits and dense target distributions."""

    __slots__ = ("axis", "reduction")
    name = "cross_entropy"

    def __init__(
        self,
        *,
        axis: int = -1,
        reduction: Reduction = "mean",
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "reduction", reduction)

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        axis = self.axis
        reduction = self.reduction
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)
        logits, targets = broadcast_tensors(logits, targets)
        _validate_distributions(targets, axis)
        output_shape = logits.shape[:axis] + logits.shape[axis + 1:]
        dtype = result_dtype(logits.dtype, targets, division=True)
        result_shape = output_shape if reduction == "none" else (1,)
        storage = execute_cross_entropy(
            logits,
            targets,
            axis,
            reduction=reduction,
            dtype=dtype,
            output_shape=result_shape,
        )
        if storage is not None:
            return Tensor._from_owned_storage(storage, dtype=dtype, shape=result_shape)
        log_probabilities = LogSoftmax(axis=axis).forward(logits)
        _, _, groups = reduction_groups(logits, axis, keepdims=False)

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

        if reduction == "none":
            return Tensor(losses, dtype=dtype, shape=output_shape)
        if reduction == "mean":
            total = _stable_float_mean(losses)
        else:
            total = _stable_float_sum(losses)
        return Tensor([total], dtype=dtype, shape=(1,))

    def backward(
        self,
        grad: Tensor,
        *inputs: Tensor,
        needs_input_grad: tuple[bool, ...],
    ) -> List[Optional[Tensor]]:
        logits, targets = inputs
        need_logits, need_targets = needs_input_grad
        axis = self.axis
        reduction = self.reduction
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)

        expanded_logits, expanded_targets = broadcast_tensors(logits, targets)
        _validate_distributions(expanded_targets, axis)
        output_shape = (
            expanded_logits.shape[:axis]
            + expanded_logits.shape[axis + 1:]
        )
        expected_shape = output_shape if reduction == "none" else (1,)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape "
                f"{expected_shape}"
            )
        accelerated = execute_cross_entropy_gradient(
            grad,
            expanded_logits,
            expanded_targets,
            axis,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
        if accelerated is not None:
            logits_storage, targets_storage = accelerated
            expanded_shape = expanded_logits.shape
            return [
                sum_to_shape(
                    Tensor._from_owned_storage(
                        logits_storage,
                        dtype=grad.dtype,
                        shape=expanded_shape,
                    ),
                    logits.shape,
                )
                if logits_storage is not None
                else None,
                sum_to_shape(
                    Tensor._from_owned_storage(
                        targets_storage,
                        dtype=grad.dtype,
                        shape=expanded_shape,
                    ),
                    targets.shape,
                )
                if targets_storage is not None
                else None,
            ]

        probabilities = (
            Softmax(axis=axis).forward(expanded_logits)
            if need_logits
            else None
        )
        log_probabilities = (
            LogSoftmax(axis=axis).forward(expanded_logits)
            if need_targets
            else None
        )
        _, _, groups = reduction_groups(
            expanded_logits,
            axis,
            keepdims=False,
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
            if need_targets:
                for index in group:
                    targets_gradient[index] = (
                        -upstream * log_probabilities._data[index]
                    )
            if not need_logits:
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

        expanded_shape = expanded_logits.shape
        return [
            sum_to_shape(
                Tensor(logits_gradient, dtype=grad.dtype, shape=expanded_shape),
                logits.shape,
            )
            if need_logits
            else None,
            sum_to_shape(
                Tensor(targets_gradient, dtype=grad.dtype, shape=expanded_shape),
                targets.shape,
            )
            if need_targets
            else None,
        ]

    def backward_graph(
        self,
        grad,
        *inputs,
        needs_input_grad: tuple[bool, ...],
    ):
        """Build a differentiable cross-entropy VJP."""
        from .log_softmax import _log_softmax_vjp
        from .reshape import reshape

        logits, targets = inputs
        axis = self.axis
        reduction = self.reduction
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
        need_logits, need_targets = needs_input_grad
        logits_derivative = (
            _log_softmax_vjp(-expanded_targets, logits, axis)
            if need_logits
            else None
        )
        targets_derivative = (
            -log_softmax(logits, axis=axis) + expanded_targets * 0.0
            if need_targets
            else None
        )

        if reduction == "none":
            upstream = reshape(grad, keepdims_shape(logits.shape, axis))
        else:
            upstream = grad
            if reduction == "mean":
                sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
                sample_count = Shape.from_iterable(sample_shape).size
                if sample_count:
                    upstream = upstream / sample_count

        return [
            sum_to_shape_graph(upstream * logits_derivative, logits.shape)
            if need_logits
            else None,
            sum_to_shape_graph(upstream * targets_derivative, targets.shape)
            if need_targets
            else None,
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
        logits_variable = (
            logits
            if isinstance(logits, Variable)
            else Variable(logits_tensor, requires_grad=False)
        )
        targets_variable = (
            prepared_targets
            if isinstance(prepared_targets, Variable)
            else Variable(_tensor(prepared_targets), requires_grad=False)
        )
        operation = CrossEntropy(axis=axis, reduction=reduction)
        return Variable._record_operation(
            operation.forward(logits_variable.data, targets_variable.data),
            operation,
            (logits_variable, targets_variable),
        )

    operation = CrossEntropy(axis=axis, reduction=reduction)
    return operation.forward(logits_tensor, _tensor(prepared_targets))


__all__ = ["CrossEntropy", "Reduction", "cross_entropy"]
