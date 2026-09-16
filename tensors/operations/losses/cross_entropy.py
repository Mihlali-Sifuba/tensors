"""Numerically stable multiclass cross-entropy."""

from __future__ import annotations
import math
from typing import TYPE_CHECKING, Any, List, Literal, Optional, overload
from tensors._typing import TensorData, TensorLike, TensorResult
from tensors.backend import (
    execute_cross_entropy,
    execute_cross_entropy_gradient,
    execute_one_hot_targets,
    execute_validate_distributions,
)
from tensors.dtype import result_dtype
from tensors.operations._gradient_shaping import sum_to_shape, sum_to_shape_graph
from tensors.shape import Shape
from tensors.operations.base import Operation
from tensors.tensor import Tensor
from tensors.graph.expression import as_tensor_operand
from tensors.utils.broadcasting import broadcast_tensors
from tensors.utils.reductions import keepdims_shape
from tensors.operations.normalization.log_softmax import log_softmax
from tensors.operations.normalization.softmax import _normalize_axis

if TYPE_CHECKING:
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable
Reduction = Literal["none", "mean", "sum"]


def _validate_reduction(reduction: str) -> None:
    if not isinstance(reduction, str):
        raise TypeError("reduction must be a string")
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _tensor(value: Any) -> Tensor:
    from tensors.variable import Variable

    if isinstance(value, Variable):
        return value.data
    return as_tensor_operand(value)


def _one_hot_targets(logits: Tensor, targets: Tensor, axis: int) -> Tensor:
    """Expand one class index per sample into dense target distributions."""
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
    scalar_target = sample_shape == () and targets.size == 1
    if targets.shape != sample_shape and (not scalar_target):
        raise ValueError(
            f"Class-index target shape {targets.shape} does not match logits sample shape {sample_shape}"
        )
    accelerated = execute_one_hot_targets(logits, targets, axis)
    return Tensor._from_owned_storage(
        accelerated, dtype=logits.dtype, shape=logits.shape
    )


def _targets_are_class_indices(logits: Tensor, targets: Tensor, axis: int) -> bool:
    """Whether ``targets`` names one class per sample.

    The decision is the values' to make, not the caller's: a target
    shaped like the logits is already a distribution, and one shaped
    like a sample is read as class indices unless it is floating,
    broadcasts over the class axis, and either holds a non-integral
    value or forms a valid distribution when expanded. Anything else
    must broadcast to the logits, or it fits neither reading.
    """
    sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
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
            (not math.isfinite(value) or not value.is_integer() for value in values)
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
            f"Target shape {targets.shape} is neither class-index shaped nor broadcastable to logits shape {logits.shape}"
        ) from exc
    if target_shape != logits.shape:
        raise ValueError(
            f"Target shape {targets.shape} cannot broadcast to logits shape {logits.shape}"
        )
    return False


def _dense_targets(logits: Tensor, targets: Tensor, axis: int) -> tuple[Tensor, bool]:
    """Return dense targets and whether they came from class indices.

    This is the operation's own preparation, so it runs from the values
    each pass is handed rather than from anything remembered between
    passes, and forward and backward reach the same reading.
    """
    if _targets_are_class_indices(logits, targets, axis):
        return (_one_hot_targets(logits, targets, axis), True)
    return (targets, False)


def _validate_distributions(targets: Tensor, axis: int) -> None:
    execute_validate_distributions(targets, axis)


def _reject_variable_class_indices() -> None:
    """Refuse class indices that were supplied as a Variable.

    Class indices name a choice rather than a quantity, so the public
    contract does not let a Variable stand for them however its
    gradient flag is set. Whether they are class indices is the values'
    to say, so this is raised wherever that reading is reached.
    """
    raise TypeError("Class-index targets cannot be differentiable Variables")


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
        object.__setattr__(self, "targets_from_variable", targets_from_variable)

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
        output_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
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
        return Tensor._from_owned_storage(storage, dtype=dtype, shape=result_shape)

    def backward(
        self, grad: Tensor, *inputs: Tensor, needs_input_grad: tuple[bool, ...]
    ) -> List[Optional[Tensor]]:
        logits, targets = inputs
        need_logits, need_targets = needs_input_grad
        axis = self.axis
        reduction = self.reduction
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits, axis)
        target_shape = targets.shape
        targets, from_class_indices = _dense_targets(logits, targets, axis)
        if from_class_indices and (need_targets or self.targets_from_variable):
            _reject_variable_class_indices()
        expanded_logits, expanded_targets = broadcast_tensors(logits, targets)
        _validate_distributions(expanded_targets, axis)
        output_shape = expanded_logits.shape[:axis] + expanded_logits.shape[axis + 1 :]
        expected_shape = output_shape if reduction == "none" else (1,)
        if grad.shape != expected_shape:
            raise ValueError(
                f"Gradient shape {grad.shape} does not match output shape {expected_shape}"
            )
        accelerated = execute_cross_entropy_gradient(
            grad,
            expanded_logits,
            expanded_targets,
            axis,
            reduction=reduction,
            needs_input_grad=needs_input_grad,
        )
        logits_storage, targets_storage = accelerated
        expanded_shape = expanded_logits.shape
        return [
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        logits_storage, dtype=grad.dtype, shape=expanded_shape
                    ),
                    logits.shape,
                )
                if logits_storage is not None
                else None
            ),
            (
                sum_to_shape(
                    Tensor._from_owned_storage(
                        targets_storage, dtype=grad.dtype, shape=expanded_shape
                    ),
                    target_shape,
                )
                if targets_storage is not None
                else None
            ),
        ]

    def backward_graph(self, grad, *inputs, needs_input_grad: tuple[bool, ...]):
        """Build a differentiable cross-entropy VJP."""
        from tensors.operations.normalization.log_softmax import _log_softmax_vjp
        from tensors.operations.manipulation.reshape import reshape
        from tensors.variable import Variable

        logits, targets = inputs
        need_logits, need_targets = needs_input_grad
        axis = self.axis
        reduction = self.reduction
        if isinstance(axis, bool) or not isinstance(axis, int):
            raise TypeError("cross_entropy axis must be an integer")
        _validate_reduction(reduction)
        axis = _normalize_axis(logits.data, axis)
        dense, from_class_indices = _dense_targets(logits.data, targets.data, axis)
        if from_class_indices:
            if need_targets or self.targets_from_variable:
                _reject_variable_class_indices()
            targets = Variable(dense, requires_grad=False)
        ones = Tensor([1.0] * logits.size, dtype=targets.dtype, shape=logits.shape)
        expanded_targets = targets * ones
        logits_derivative = (
            _log_softmax_vjp(-expanded_targets, logits, axis) if need_logits else None
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
                sample_shape = logits.shape[:axis] + logits.shape[axis + 1 :]
                sample_count = Shape.from_iterable(sample_shape).size
                if sample_count:
                    upstream = upstream / sample_count
        return [
            (
                sum_to_shape_graph(upstream * logits_derivative, logits.shape)
                if need_logits
                else None
            ),
            (
                sum_to_shape_graph(upstream * targets_derivative, targets.shape)
                if need_targets
                else None
            ),
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
    from tensors.graph.expression import apply_operation, as_graph_operand
    from tensors.graph.node import VariableNode
    from tensors.variable import Variable

    _validate_reduction(reduction)
    if isinstance(axis, bool) or not isinstance(axis, int):
        raise TypeError("cross_entropy axis must be an integer")
    structural = isinstance(logits, VariableNode) or isinstance(targets, VariableNode)
    if isinstance(targets, Variable) and (not structural):
        logits_tensor = _tensor(logits)
        if _targets_are_class_indices(
            logits_tensor, targets.data, _normalize_axis(logits_tensor, axis)
        ):
            _reject_variable_class_indices()
    operation = CrossEntropy(
        axis=axis,
        reduction=reduction,
        targets_from_variable=isinstance(targets, Variable),
    )
    if structural:
        return apply_operation(
            operation, (as_graph_operand(logits), as_graph_operand(targets))
        )
    if isinstance(logits, Variable) or isinstance(targets, Variable):
        return Variable._apply_operation(
            operation,
            (
                (
                    logits
                    if isinstance(logits, Variable)
                    else Variable(_tensor(logits), requires_grad=False)
                ),
                (
                    targets
                    if isinstance(targets, Variable)
                    else Variable(_tensor(targets), requires_grad=False)
                ),
            ),
        )
    return operation.forward(_tensor(logits), _tensor(targets))


__all__ = ["CrossEntropy", "Reduction", "cross_entropy"]
