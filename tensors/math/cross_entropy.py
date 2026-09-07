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
from ..graph.expression import as_tensor_operand
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
    from ..graph.node import VariableNode
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
    return as_tensor_operand(value)


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


def _targets_are_class_indices(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> bool:
    """Whether ``targets`` names one class per sample.

    The decision is the values' to make, not the caller's: a target
    shaped like the logits is already a distribution, and one shaped
    like a sample is read as class indices unless it is floating,
    broadcasts over the class axis, and either holds a non-integral
    value or forms a valid distribution when expanded. Anything else
    must broadcast to the logits, or it fits neither reading.
    """
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1:]
    scalar_target = sample_shape == () and targets.size == 1

    if targets.shape == logits.shape:
        return False

    if targets.shape == sample_shape or scalar_target:
        if targets.dtype.kind != "floating":
            return True
        try:
            target_shape = targets.shape.broadcast_with(logits.shape)
        except ValueError:
            return True
        if target_shape != logits.shape:
            return True
        values = [float(value) for value in targets._data]
        if any(
            not math.isfinite(value) or not value.is_integer()
            for value in values
        ):
            return False
        _, expanded_targets = broadcast_tensors(logits, targets)
        try:
            _validate_distributions(expanded_targets, axis)
        except ValueError:
            return True
        return False

    try:
        target_shape = targets.shape.broadcast_with(logits.shape)
    except ValueError as exc:
        raise ValueError(
            f"Target shape {targets.shape} is neither class-index shaped "
            f"nor broadcastable to logits shape {logits.shape}"
        ) from exc
    if target_shape != logits.shape:
        raise ValueError(
            f"Target shape {targets.shape} cannot broadcast to "
            f"logits shape {logits.shape}"
        )
    return False


def _dense_targets(
    logits: Tensor,
    targets: Tensor,
    axis: int,
) -> tuple[Tensor, bool]:
    """Return dense targets and whether they came from class indices.

    This is the operation's own preparation, so it runs from the values
    each pass is handed rather than from anything remembered between
    passes, and forward and backward reach the same reading.
    """
    if _targets_are_class_indices(logits, targets, axis):
        return _one_hot_targets(logits, targets, axis), True
    return targets, False


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


def _reject_variable_class_indices() -> None:
    """Refuse class indices that were supplied as a Variable.

    Class indices name a choice rather than a quantity, so the public
    contract does not let a Variable stand for them however its
    gradient flag is set. Whether they are class indices is the values'
    to say, so this is raised wherever that reading is reached.
    """
    raise TypeError(
        "Class-index targets cannot be differentiable Variables"
    )


class CrossEntropy(Operation):
    """Stable cross-entropy between logits and raw targets.

    The second operand is the targets as the caller wrote them: either
    one class index per sample, or dense probability distributions
    broadcastable to the logits. Which of the two a target holds is
    read from its values on every pass, so one operation replays
    against changing targets.

    ``targets_from_variable`` records how the call was written rather
    than what its values turned out to be: the graph keeps a Tensor
    constant and a caller's Variable as the same kind of leaf, so the
    public rule that class indices may not be a Variable would
    otherwise be unenforceable once the call is recorded.
    """

    __slots__ = ("axis", "reduction", "targets_from_variable")
    name = "cross_entropy"

    def __init__(
        self,
        *,
        axis: int = -1,
        reduction: Reduction = "mean",
        targets_from_variable: bool = False,
    ) -> None:
        object.__setattr__(self, "axis", axis)
        object.__setattr__(self, "reduction", reduction)
        object.__setattr__(
            self, "targets_from_variable", targets_from_variable
        )

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        axis = self.axis
        reduction = self.reduction
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)
        targets, from_class_indices = _dense_targets(logits, targets, axis)
        if from_class_indices and self.targets_from_variable:
            _reject_variable_class_indices()
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

        # The gradient is summed back to the operand's own shape, which
        # is the raw target's, so it is read before preparation.
        target_shape = targets.shape
        targets, from_class_indices = _dense_targets(logits, targets, axis)
        if from_class_indices and (need_targets or self.targets_from_variable):
            _reject_variable_class_indices()

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
                    target_shape,
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
                target_shape,
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

        from ..variable import Variable

        logits, targets = inputs
        need_logits, need_targets = needs_input_grad
        axis = self.axis
        reduction = self.reduction
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits.data, axis)

        dense, from_class_indices = _dense_targets(
            logits.data, targets.data, axis
        )
        if from_class_indices:
            if need_targets or self.targets_from_variable:
                _reject_variable_class_indices()
            # One class per sample is a constant here, so the expansion
            # enters the derivative as a non-gradient value.
            targets = Variable(dense, requires_grad=False)

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
    logits: VariableNode,
    targets: TensorLike | VariableNode,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> VariableNode: ...


@overload
def cross_entropy(
    logits: TensorLike,
    targets: VariableNode,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> VariableNode: ...


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
    logits: TensorLike | VariableNode,
    targets: TensorLike | VariableNode,
    *,
    axis: int = -1,
    reduction: Reduction = "mean",
) -> TensorResult | VariableNode:
    """Return stable multiclass cross-entropy from unnormalized ``logits``.

    ``targets`` may contain class indices with the class axis removed, or
    dense probability distributions broadcastable to the logits shape.
    In an otherwise ambiguous shape, an integer dtype selects class indices
    while a floating probability distribution selects dense targets.

    Which reading applies is decided from the values, so the operation
    makes it when it runs and this function only says what to record: a
    graph value in either position applies the loss through the graph,
    with the raw targets as the second operand.
    """
    from ..graph.expression import apply_operation, as_graph_operand
    from ..graph.node import VariableNode
    from ..variable import Variable

    _validate_reduction(reduction)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("cross_entropy axis must be an integer")

    structural = isinstance(logits, VariableNode) or isinstance(
        targets, VariableNode
    )
    if isinstance(targets, Variable) and not structural:
        # A Variable target must be a distribution. Reading it needs the
        # logits' shape, so the guard applies wherever that shape exists
        # and otherwise waits for the pass that has it.
        logits_tensor = _tensor(logits)
        if _targets_are_class_indices(
            logits_tensor,
            targets.data,
            _normalize_axis(logits_tensor, axis),
        ):
            _reject_variable_class_indices()

    # A vertex hides the logits' rank, so the reading that decides the
    # rule above cannot be reached yet. Recording how the call was
    # written carries it to the pass that can.
    operation = CrossEntropy(
        axis=axis,
        reduction=reduction,
        targets_from_variable=isinstance(targets, Variable),
    )

    if structural:
        return apply_operation(
            operation,
            (as_graph_operand(logits), as_graph_operand(targets)),
        )

    if isinstance(logits, Variable) or isinstance(targets, Variable):
        return Variable._apply_operation(
            operation,
            (
                logits
                if isinstance(logits, Variable)
                else Variable(_tensor(logits), requires_grad=False),
                targets
                if isinstance(targets, Variable)
                else Variable(_tensor(targets), requires_grad=False),
            ),
        )

    return operation.forward(_tensor(logits), _tensor(targets))


__all__ = ["CrossEntropy", "Reduction", "cross_entropy"]
